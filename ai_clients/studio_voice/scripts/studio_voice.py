# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
import os
import sys
import grpc
import time
import soundfile as sf
import numpy as np
from typing import Iterator
import logging
from functools import wraps

# Add parent directories to path to enable imports
script_dir = os.path.dirname(os.path.abspath(__file__))
interface_dir = os.path.join(script_dir, '..', 'interfaces')
sys.path.insert(0, interface_dir)

# Importing gRPC compiler auto-generated studiovoice library
from studio_voice import studiovoice_pb2, studiovoice_pb2_grpc  # noqa: E402

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# gRPC error handling constants
MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 32  # seconds
BACKOFF_MULTIPLIER = 2.0
GRPC_DEADLINE = 300  # 5 minutes for streaming operations


def retry_with_backoff(func):
    """Decorator to retry gRPC calls with exponential backoff"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        backoff = INITIAL_BACKOFF
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except grpc.RpcError as e:
                last_exception = e
                # Only retry on specific recoverable errors
                if e.code() in [
                    grpc.StatusCode.UNAVAILABLE,
                    grpc.StatusCode.RESOURCE_EXHAUSTED,
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    grpc.StatusCode.INTERNAL,  # Retry on server errors too
                ]:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(f"Attempt {attempt + 1} failed with {e.code()}: {e.details()}. Retrying in {backoff}s...")
                        time.sleep(backoff)
                        backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
                    else:
                        logger.error(f"All {MAX_RETRIES} retry attempts failed. Last error: {e.code()} - {e.details()}")
                        raise
                else:
                    # Don't retry on non-recoverable errors
                    logger.error(f"Non-recoverable gRPC error: {e.code()} - {e.details()}")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        
        if last_exception:
            raise last_exception
    
    return wrapper


def create_channel_with_options(target: str, credentials=None):
    """Create a gRPC channel with optimal settings for streaming"""
    # Channel options for better reliability
    channel_options = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ('grpc.keepalive_time_ms', 30000),  # Send keepalive every 30s
        ('grpc.keepalive_timeout_ms', 10000),  # Wait 10s for keepalive response
        ('grpc.keepalive_permit_without_calls', True),  # Allow keepalive without active calls
        ('grpc.http2.max_pings_without_data', 0),  # Allow unlimited pings
        ('grpc.enable_retries', True),  # Enable built-in retries
        ('grpc.service_config', '{"loadBalancingPolicy":"round_robin"}'),
    ]
    
    if credentials:
        return grpc.secure_channel(target, credentials, options=channel_options)
    else:
        return grpc.insecure_channel(target, options=channel_options)


def validate_server_connectivity(channel: grpc.Channel) -> bool:
    """Check if the server is reachable before sending data"""
    try:
        # Use a small timeout for connectivity check
        state = grpc.channel_ready_future(channel).result(timeout=10)
        logger.info("Server connectivity check passed")
        return True
    except Exception as e:
        logger.error(f"Server connectivity check failed: {e}")
        return False



def read_file_content(file_path: os.PathLike) -> None:
    """Function to read file content as bytes.

    Args:
      file_path: Path to input file
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Exiting.")

    with open(file_path, "rb") as file:
        return file.read()


def generate_request_for_inference(
    input_filepath: os.PathLike, model_type: str, sample_rate: int, streaming: bool
) :
    """Generator to produce the request data stream

    Args:
      input_filepath: Path to input file
      model_type: Studio Voice model type to infer
      sample_rate: Input audio sample rate
      streaming: Enables grpc streaming mode
    """
    if streaming:
        """
        Input audio chunk is generated based on model type and sample rate,
        1) High quality models require 6sec input
        2) Low latency models require 10ms input chunk
        """
        input_audio, sample_rate_file = sf.read(input_filepath)
        input_audio = input_audio.astype(np.float32)  # Convert to float32
        input_size_in_ms = 10 if (model_type == "48k-ll") else 6000
        samples_per_ms = sample_rate // 1000
        input_float_size = int(input_size_in_ms * samples_per_ms)

        pad_length = (input_float_size - len(input_audio) % input_float_size) % input_float_size
        if pad_length > 0:
            input_audio = np.pad(input_audio, (0, pad_length), "constant")

        print(f"Streaming audio with total samples: {len(input_audio)}, chunk size: {input_float_size} samples.")
        for i in range(0, len(input_audio), input_float_size):
            data = input_audio[i : i + input_float_size]
            yield studiovoice_pb2.EnhanceAudioRequest(audio_stream_data=data.tobytes())
    else:
        DATA_CHUNKS = 64 * 1024  # bytes, we send the wav file in 64KB chunks
        with open(input_filepath, "rb") as fd:
            while True:
                buffer = fd.read(DATA_CHUNKS)
                if buffer == b"":
                    break
                yield studiovoice_pb2.EnhanceAudioRequest(audio_stream_data=buffer)


def write_output_file_from_response(
    response_iter: Iterator[studiovoice_pb2.EnhanceAudioResponse],
    output_filepath: os.PathLike,
    sample_rate: int,
    streaming: bool,
) -> int:
    """Function to write the output file from the incoming gRPC data stream with error handling.

    Args:
      response_iter: Responses from the server to write into output file
      output_filepath: Path to output file
      sample_rate: Input audio sample rate
      streaming: Enables grpc streaming mode
      
    Returns:
      Number of response chunks received
    """
    response_count = 0
    try:
        if streaming:
            output_audio = []
            for response in response_iter:
                try:
                    if response.HasField("audio_stream_data"):
                        response_count += 1
                        output_audio.append(np.frombuffer(response.audio_stream_data, np.float32))
                except Exception as e:
                    logger.error(f"Error processing response chunk {response_count + 1}: {e}")
                    raise
            
            if not output_audio:
                raise RuntimeError("No audio data received from server")
            
            # Write combined audio data
            combined_audio = np.hstack(output_audio)
            sf.write(output_filepath, combined_audio, sample_rate)
            logger.info(f"Successfully wrote {response_count} audio chunks to output file")
        else:
            with open(output_filepath, "wb") as fd:
                for response in response_iter:
                    response_count += 1
                    if response.HasField("audio_stream_data"):
                        fd.write(response.audio_stream_data)
            logger.info(f"Successfully wrote audio data to output file")
                
        return response_count
        
    except Exception as e:
        logger.error(f"Failed to write output file after {response_count} chunks: {e}")
        # Clean up incomplete output file
        if os.path.exists(output_filepath):
            try:
                os.remove(output_filepath)
                logger.info("Cleaned up incomplete output file")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up output file: {cleanup_error}")
        raise


def parse_args() -> None:
    """
    Parse command-line arguments using argparse.
    """
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Process wav audio files using gRPC and apply studio-voice."
    )
    parser.add_argument(
        "--preview-mode",
        action="store_true",
        help="Flag to send request to preview NVCF NIM server on "
        "https://build.nvidia.com/nvidia/studiovoice/api. ",
    )
    parser.add_argument(
        "--ssl-mode",
        type=str,
        help="Flag to set SSL mode, default is None",
        default=None,
        choices=["MTLS", "TLS"],
    )
    parser.add_argument(
        "--ssl-key",
        type=str,
        default=None,
        help="The path to ssl private key.",
    )
    parser.add_argument(
        "--ssl-cert",
        type=str,
        default=None,
        help="The path to ssl certificate chain.",
    )
    parser.add_argument(
        "--ssl-root-cert",
        type=str,
        default=None,
        help="The path to ssl root certificate.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="127.0.0.1:8001",
        help="IP:port of gRPC service, when hosted locally. "
        "Use grpc.nvcf.nvidia.com:443 when hosted on NVCF.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="../assets/studio_voice_48k_input.wav",
        help="The path to the input audio file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="studio_voice_48k_output.wav",
        help="The path for the output audio file.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="NGC API key required for authentication, "
        "utilized when using TRY API ignored otherwise",
    )
    parser.add_argument(
        "--function-id",
        type=str,
        help="NVCF function ID for the service, utilized when using TRY API ignored otherwise",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Flag to enable grpc streaming mode. ",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        help="Studio Voice model type, default is 48k-hq. ",
        default="48k-hq",
        choices=["48k-hq", "48k-ll", "16k-hq"],
    )
    return parser.parse_args()


def process_request(
    channel: any,
    input_filepath: os.PathLike,
    output_filepath: os.PathLike,
    model_type: str,
    sample_rate: int,
    streaming: bool,
    request_metadata: dict = None,
) -> None:
    """Function to process gRPC request with proper error handling

    Args:
      channel: gRPC channel for server client communication
      input_filepath: Path to input file
      output_filepath: Path to output file
      model_type: Studio Voice model type to infer
      sample_rate: Input audio sample rate
      streaming: Enables grpc streaming mode
      request_metadata: Credentials to process request
    """
    # Validate server connectivity before processing
    if not validate_server_connectivity(channel):
        raise RuntimeError("Unable to establish connection to the server. Please check the server address and network connectivity.")
    
    @retry_with_backoff
    def execute_streaming_request():
        try:
            stub = studiovoice_pb2_grpc.StudioVoiceStub(channel)
            start_time = time.time()
            
            # Create request generator
            request_generator = generate_request_for_inference(
                input_filepath=input_filepath,
                model_type=model_type,
                sample_rate=sample_rate,
                streaming=streaming,
            )
            
            # Execute the RPC call with deadline
            responses = stub.EnhanceAudio(
                request_generator,
                metadata=request_metadata,
                timeout=GRPC_DEADLINE,
            )
            
            # Process responses with proper error handling
            response_count = write_output_file_from_response(
                response_iter=responses,
                output_filepath=output_filepath,
                sample_rate=sample_rate,
                streaming=streaming,
            )
            
            end_time = time.time()
            
            if streaming:
                avg_latency = (end_time - start_time) / response_count if response_count > 0 else 0
                logger.info(f"Average latency per request: {avg_latency*1000:.2f}ms")
                logger.info(f"Processed {response_count} chunks.")
            
            logger.info(
                f"Function invocation completed in {end_time-start_time:.2f}s, "
                "the output file is generated."
            )
            
        except grpc.RpcError as e:
            error_msg = f"gRPC Error ({e.code()}): {e.details()}"
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"Unexpected error during processing: {e}"
            logger.error(error_msg)
            raise
    
    # Execute with retry logic
    execute_streaming_request()


def main():
    """
    Main client function
    """
    args = parse_args()
    run_client(**vars(args))


def run_client(**kwargs):
    """
    Core client function that can be called from other scripts or command line.
    """
    args = argparse.Namespace(**kwargs)
    streaming = args.streaming
    model_type = args.model_type
    logger.info(f"Streaming mode set to {streaming}")
    sample_rate = 48000
    if model_type == "16k-hq":
        sample_rate = 16000
    logger.info(f"Sample Rate: {sample_rate}")
    input_filepath = getattr(args, "input", None)
    output_filepath = getattr(args, "output", None)

    # Check if input file path exists
    if os.path.isfile(input_filepath):
        logger.info(f"The file '{input_filepath}' exists. Proceeding with processing.")
    else:
        raise FileNotFoundError(f"The file '{input_filepath}' does not exist. Exiting.")

    # Check the sample rate of the input audio file
    input_info = sf.info(input_filepath)
    input_sample_rate = input_info.samplerate
    logger.info(f"Input file sample rate: {input_sample_rate}")

    # Check if the input file's sample rate matches the expected sample rate
    if input_sample_rate != sample_rate:
        # This check is useful for command-line, but for API calls, we resample.
        # Let's print a warning instead of raising an error.
        logger.warning(f"Sample rate mismatch: model expects {sample_rate}, but input is {input_sample_rate}. The API route should handle resampling.")

    preview_mode = getattr(args, "preview_mode", False)
    ssl_mode = getattr(args, "ssl_mode", None)

    if preview_mode:
        if ssl_mode != "TLS":
            # Preview mode only supports TLS mode
            ssl_mode = "TLS"
            logger.info("--ssl-mode is set as TLS, since preview_mode is enabled.")
        if getattr(args, "ssl_root_cert", None):
            raise RuntimeError("Preview mode does not support custom root certificate.")

    if ssl_mode is not None:
        request_metadata = None
        root_certificates = None
        if ssl_mode == "MTLS":
            ssl_key = getattr(args, "ssl_key", None)
            ssl_cert = getattr(args, "ssl_cert", None)
            ssl_root_cert = getattr(args, "ssl_root_cert", None)
            if not (ssl_key and ssl_cert and ssl_root_cert):
                raise RuntimeError("If --ssl-mode is MTLS, --ssl-key, --ssl-cert and --ssl-root-cert are required.")
            private_key = read_file_content(args.ssl_key)
            certificate_chain = read_file_content(args.ssl_cert)
            root_certificates = read_file_content(args.ssl_root_cert)
            channel_credentials = grpc.ssl_channel_credentials(
                root_certificates=root_certificates,
                private_key=private_key,
                certificate_chain=certificate_chain,
            )
        else:
            # Running with NVCF
            if args.preview_mode:
                request_metadata = (
                    ("authorization", "Bearer {}".format(args.api_key)),
                    ("function-id", args.function_id),
                )
                channel_credentials = grpc.ssl_channel_credentials()
            # Running TLS mode, without NVCF
            else:
                if not (args.ssl_root_cert):
                    raise RuntimeError("If --ssl-mode is TLS, --ssl-root-cert is required.")
                root_certificates = read_file_content(args.ssl_root_cert)
                channel_credentials = grpc.ssl_channel_credentials(
                    root_certificates=root_certificates
                )

        # Use the new channel creation function with options
        channel = create_channel_with_options(args.target, channel_credentials)
        try:
            process_request(
                channel=channel,
                input_filepath=input_filepath,
                output_filepath=output_filepath,
                model_type=model_type,
                sample_rate=sample_rate,
                streaming=streaming,
                request_metadata=request_metadata,
            )
        finally:
            channel.close()
    else:
        # Use the new channel creation function with options
        channel = create_channel_with_options(args.target)
        try:
            process_request(
                channel=channel,
                input_filepath=input_filepath,
                output_filepath=output_filepath,
                model_type=model_type,
                sample_rate=sample_rate,
                streaming=streaming,
            )
        finally:
            channel.close()


if __name__ == "__main__":
    main()
