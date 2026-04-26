#!/usr/bin/env python3

import sys
import configparser
import socket
import struct
import numpy as np
from collections import deque

# load preferences file
config = configparser.ConfigParser()
config.read("feed_my_wled.conf")

def parse_wled_targets(config_obj):
    """Parse WLED targets from config.

    Preferred format:
      WLED_TARGETS = 192.168.1.25, 192.168.1.26:11988

    Backward compatible fallback:
      WLED_IP_ADDRESS + WLED_UDP_PORT
    """
    default_port = config_obj.getint("WLED", "WLED_UDP_PORT", fallback=11988)
    raw_targets = config_obj.get("WLED", "WLED_TARGETS", fallback="").strip()
    parsed_targets = []

    if raw_targets:
        for entry in raw_targets.split(","):
            target = entry.strip()
            if not target:
                continue

            if ":" in target:
                host, port_str = target.rsplit(":", 1)
                host = host.strip()
                port = int(port_str.strip())
            else:
                host = target
                port = default_port

            parsed_targets.append((host, port))
    else:
        host = config_obj.get("WLED", "WLED_IP_ADDRESS")
        parsed_targets.append((host, default_port))

    if not parsed_targets:
        raise ValueError("No WLED targets configured.")

    return parsed_targets

#load preferences
WLED_TARGETS = parse_wled_targets(config)
sample_rate = config.getint("Audio", "sample_rate")
buffer_size = config.getint("Audio", "buffer_size")
chunk_size = config.getint("Audio", "chunk_size")
enable_buffer_delay = config.getboolean("Audio", "enable_buffer_delay", fallback=False)

# Pro-style dynamics settings for cleaner audio-reactive behavior.
noise_gate_level = config.getfloat("Processing", "noise_gate_level", fallback=900.0)
gate_hysteresis = config.getfloat("Processing", "gate_hysteresis", fallback=0.8)
level_attack = config.getfloat("Processing", "level_attack", fallback=0.35)
level_release = config.getfloat("Processing", "level_release", fallback=0.08)
level_curve = config.getfloat("Processing", "level_curve", fallback=1.6)
fft_floor = config.getfloat("Processing", "fft_floor", fallback=0.08)
fft_curve = config.getfloat("Processing", "fft_curve", fallback=1.4)
fft_agc_attack = config.getfloat("Processing", "fft_agc_attack", fallback=0.25)
fft_agc_release = config.getfloat("Processing", "fft_agc_release", fallback=0.02)

# def vars
previous_smoothed_level = 0.0
ring_buffer = deque(maxlen=max(1, buffer_size // chunk_size))  # Ring buffer of delayed chunks
gate_is_open = False
fft_agc_reference = 1.0

# create socket
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

## functions
def apply_noise_gate(level):
    """Gate low-level noise using hysteresis to avoid chatter."""
    global gate_is_open

    open_threshold = noise_gate_level
    close_threshold = noise_gate_level * gate_hysteresis

    if gate_is_open:
        if level < close_threshold:
            gate_is_open = False
    else:
        if level > open_threshold:
            gate_is_open = True

    return level if gate_is_open else 0.0

def shape_fft_values(fft_magnitudes):
    """Apply adaptive gain, floor suppression, and curve shaping to FFT bands."""
    global fft_agc_reference

    if fft_magnitudes is None or len(fft_magnitudes) == 0:
        return np.zeros(16, dtype=np.uint8)

    max_magnitude = float(np.max(fft_magnitudes))
    if max_magnitude > fft_agc_reference:
        fft_agc_reference = ((1.0 - fft_agc_attack) * fft_agc_reference) + (fft_agc_attack * max_magnitude)
    else:
        fft_agc_reference = ((1.0 - fft_agc_release) * fft_agc_reference) + (fft_agc_release * max_magnitude)

    fft_agc_reference = max(fft_agc_reference, 1.0)

    normalized = np.clip(fft_magnitudes / fft_agc_reference, 0.0, 1.0)
    if 0.0 < fft_floor < 1.0:
        normalized = np.where(normalized <= fft_floor, 0.0, (normalized - fft_floor) / (1.0 - fft_floor))

    normalized = np.power(normalized, fft_curve)
    return (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)

# calculating fft
def calculate_fft(audio_chunk, sample_rate):
    """
    Calc FFT for a Audioblock
    :param audio_chunk: Audiodata as Byte-Array.
    :param sample_rate: Samplerate of Audiodata.
    :return: Tuple (FFT-Ergebnisse for 16 Frequency bands, additional peaks).
    """
    try:
        if not audio_chunk:
            return None, 0, 0, 0, 0

        # Ensure a valid int16 byte length (2 bytes/sample)
        if len(audio_chunk) % 2 != 0:
            audio_chunk = audio_chunk[:-1]

        # Convert to a numpy-Array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16)

        if audio_data.size == 0:
            return None, 0, 0, 0, 0

        # Calc Peak (Raw Level and Peak Level)
        raw_level = np.mean(np.abs(audio_data))
        peak_level = int((np.max(np.abs(audio_data)) / 32767) * 255)

        # Calc FFT on a windowed signal for cleaner band stability.
        windowed = audio_data.astype(np.float32) * np.hanning(len(audio_data))
        fft_result = np.abs(np.fft.rfft(windowed))

        # Select first 16 frequency bands as raw magnitudes.
        fft_values = fft_result[:16]

        # Find dominating frequency
        freq_index = np.argmax(fft_result)
        fft_peak_frequency = freq_index * (sample_rate / len(audio_data))

        # sum of fft magnitudes
        fft_magnitude_sum = np.sum(fft_result)

        # return calced values
        return fft_values, raw_level, peak_level, fft_magnitude_sum, fft_peak_frequency

        #error handling
    except Exception as e:
        print(f"Error at calcing FFT: {e}")
        return None, 0, 0, 0, 0

# function for creating the udp package
def create_udp_packet(fft_values, raw_level, smoothed_level, peak_level, fft_magnitude_sum, fft_peak_frequency):
    """
    Creating udp-package in a wled compatible format
    :param fft_values: fft datas for 16 frequency bands
    :param raw_level: mean of audiosource
    :param smoothed_level: smoothed level of audiosource
    :param peak_level: mac peak of audiosource
    :param fft_magnitude_sum: sum of fft magnitudes
    :param fft_peak_frequency: dominant frequency of audiosignal
    :return: formated UDP-package
    """

    # Convert Values
    peak_level = max(0, min(255, int(peak_level)))
    fft_values = list(fft_values)
    if len(fft_values) < 16:
        fft_values.extend([0] * (16 - len(fft_values)))
    fft_values = [max(0, min(255, int(v))) for v in fft_values[:16]]

    # Create package according to the WLED Protocol
    udp_packet = struct.pack('<6s2B2fBB16B2B2f',        # don't mess with that!
        b'00002',                   # Header (6 Bytes)
        0,0,                        # Gap (2 Bytes)
        float(raw_level),           # Raw Level (4 Bytes Float)
        float(smoothed_level),      # Smoothed Level (4 Bytes Float)
        peak_level,                 # Peak Level (1 Byte)
        0,                          # static 0 (1 Byte)
        *fft_values,                # FFT Result (16 Bytes)
        0,0,                        # Gap (2 Bytes)
        float(fft_magnitude_sum),   # FFT Magnitude (4 Bytes Float)
        float(fft_peak_frequency))  # FFT Major Peak (4 Bytes Float)
    return udp_packet

# main function
def stream_audio_to_wled():
    """
    Reads audiodata from stream and send analyzed fft results to WLED
    """
    global previous_smoothed_level, ring_buffer

    try:
        target_text = ", ".join(f"{host}:{port}" for host, port in WLED_TARGETS)
        print(f"Start Pipe to WLED targets: {target_text}")
        print(
            f"Processing: gate={noise_gate_level:.0f}, "
            f"hyst={gate_hysteresis:.2f}, lvl_curve={level_curve:.2f}, fft_floor={fft_floor:.2f}"
        )

        if enable_buffer_delay and ring_buffer.maxlen > 1:
            delay_ms = ((ring_buffer.maxlen - 1) * chunk_size / 2 / sample_rate) * 1000
            print(f"Delay mode enabled (~{delay_ms:.0f} ms)")
        else:
            print("Low-latency mode enabled")

        # open stream from pipe
        while True:
            # read datas from pipe
            audio_data = sys.stdin.buffer.read(chunk_size)
            if not audio_data:
                # End of input stream (pipe closed)
                break

            ring_buffer.append(audio_data)  # push data to buffer

            # combine blocks for prozessing
            combined_data = b"".join(ring_buffer)

            if enable_buffer_delay and ring_buffer.maxlen > 1:
                # Use the oldest chunk to create an intentional delay.
                analysis_chunk = combined_data[:chunk_size]
            else:
                # Use the newest chunk for minimum latency.
                analysis_chunk = audio_data

            # Calc FFT and Peaks with buffersize
            fft_result = calculate_fft(analysis_chunk, sample_rate)
            if fft_result[0] is None:
                print("Unvalid FFT-Datas, skip actual block.")
                continue

            # feed fft_result to its vars
            fft_data_raw, raw_level, peak_level, fft_magnitude_sum, fft_peak_frequency = fft_result

            gated_level = apply_noise_gate(raw_level)
            if gated_level <= 0:
                raw_level = 0.0
                peak_level = 0
                fft_data = np.zeros(16, dtype=np.uint8)
            else:
                # Curve shaping de-emphasizes low-level detail and highlights real hits.
                normalized_level = min(1.0, gated_level / 32767.0)
                raw_level = (normalized_level ** level_curve) * 32767.0
                peak_level = int((raw_level / 32767.0) * 255)
                fft_data = shape_fft_values(fft_data_raw)

            # Asymmetric attack/release envelope like stage audio processors.
            smoothing = level_attack if raw_level > previous_smoothed_level else level_release
            smoothed_level = ((1.0 - smoothing) * previous_smoothed_level) + (smoothing * raw_level)
            previous_smoothed_level = smoothed_level

            # Create UDP-Paket
            udp_packet = create_udp_packet(fft_data, raw_level, smoothed_level, peak_level, fft_magnitude_sum, fft_peak_frequency)

            # Send Package to WLED
            for target in WLED_TARGETS:
                udp_socket.sendto(udp_packet, target)

    except KeyboardInterrupt:
        print("Audiostreaming closed.")
    finally:
        udp_socket.close()

# start 
stream_audio_to_wled()
