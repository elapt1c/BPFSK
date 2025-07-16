import sounddevice as sd
import numpy as np
from scipy.fft import fft, fftfreq
import time
import collections
import sys # For sys.stdout.flush and sys.stdout.write
import os # For file saving

# --- Configuration ---
SAMPLING_RATE = 44100
CHANNEL_DURATION = 0.03
PREAMBLE_DURATION = 1.0
FREQUENCY_TOLERANCE = 100

# NOMINAL_CHANNEL_FREQUENCIES: Dynamically generated to fit 10kHz to 18kHz
NOMINAL_CHANNEL_FREQUENCIES = {}
MIN_OPERATING_FREQ_HZ = 10000
MAX_OPERATING_FREQ_HZ = 18000
NUM_TOTAL_CHANNELS = 19 # Channels 1 through 19

# Calculate the frequency step for linear spacing
FREQ_STEP = (MAX_OPERATING_FREQ_HZ - MIN_OPERATING_FREQ_HZ) / (NUM_TOTAL_CHANNELS - 1)

for i in range(1, NUM_TOTAL_CHANNELS + 1):
    NOMINAL_CHANNEL_FREQUENCIES[i] = MIN_OPERATING_FREQ_HZ + (i - 1) * FREQ_STEP

# --- Rest of the configuration remains the same ---
BLOCKSIZE_SECONDS = 0.01
BLOCKSIZE_SAMPLES = int(SAMPLING_RATE * BLOCKSIZE_SECONDS)

DETECTION_THRESHOLD_FACTOR_CHANNEL = 0.51
DETECTION_THRESHOLD_FACTOR_PREAMBLE = 0.7

MIN_CONSECUTIVE_BLOCKS_FOR_CHANNEL = max(1, int(np.ceil((CHANNEL_DURATION / BLOCKSIZE_SECONDS) * DETECTION_THRESHOLD_FACTOR_CHANNEL)))
MIN_CONSECUTIVE_BLOCKS_FOR_PREAMBLE = max(1, int(np.ceil((PREAMBLE_DURATION / BLOCKSIZE_SECONDS) * DETECTION_THRESHOLD_FACTOR_PREAMBLE)))

# TRAINING_SEQUENCE: Channels 2, then 3, then 4-19 (still part of training)
TRAINING_SEQUENCE = [2, 3] + list(range(4, 20)) 
CALIBRATION_SANE_TOLERANCE_FACTOR = 0.075

# --- Protocol Constants ---
HEADER_START_DELIMITER = 0xFE
MESSAGE_TYPE_TEXT = 0x00
MESSAGE_TYPE_FILE = 0x01
HEADER_END_DELIMITER = 0xFF

# --- Decoder State Variables ---
decoder_state = "IDLE" # Can be IDLE, CALIBRATING, READING_HEADER, RECEIVING_DATA
current_message_channels_log = []
raw_decoded_payload_bytes = [] # Stores all raw bytes after nibble assembly (header + actual data)

training_sequence_index = 0
calibrated_frequencies = {}
has_been_calibrated = False

# Variables for tracking current tone detection
current_tone_candidate_nominal_chan = None
current_tone_candidate_blocks = 0
fsm_informed_of_this_segment = False
recent_detections_nominal_chan = collections.deque(maxlen=3)
current_tone_candidate_freq_samples = []

# Global variables for byte assembly
current_high_nibble_value = None # Stores 0-15
byte_processing_state = "EXPECT_HIGH_NIBBLE" # One of: "EXPECT_HIGH_NIBBLE", "EXPECT_LOW_NIBBLE", "EXPECT_BYTE_SEPARATOR"
current_byte_channels_debug = [] # For logging channels of current byte

# File/Message specific variables
current_message_type = None
header_buffer = [] # Temporarily stores bytes while header is being read
header_parsed = False
file_metadata = { # For files, holds name/ext. For both, will hold payload_size
    "filename": "",
    "extension": "",
    "payload_size": 0 # This will be the RAW data size (text or file)
}
payload_bytes_received = 0 # Tracks how many bytes of the actual data payload (after header) have been received

# --- Helper Functions for Bytes ---
def bytes_to_int_le(byte_array):
    """Converts a little-endian byte array to an integer."""
    return int.from_bytes(byte_array, 'little')

def get_channel_map_for_find():
    """Returns the calibrated frequency map if available, otherwise the nominal map."""
    if has_been_calibrated and calibrated_frequencies and len(calibrated_frequencies) == len(NOMINAL_CHANNEL_FREQUENCIES):
        return calibrated_frequencies
    return NOMINAL_CHANNEL_FREQUENCIES

def find_closest_channel(frequency, use_nominal_map_only=False):
    """
    Finds the closest nominal channel ID for a given frequency.
    """
    if frequency is None: return None

    source_freq_map = NOMINAL_CHANNEL_FREQUENCIES
    if not use_nominal_map_only and has_been_calibrated and len(calibrated_frequencies) == len(NOMINAL_CHANNEL_FREQUENCIES):
        source_freq_map = calibrated_frequencies

    min_diff = float('inf')
    closest_channel_num = None

    for channel_num, freq_val in source_freq_map.items():
        diff = abs(frequency - freq_val)
        if diff < min_diff and diff < FREQUENCY_TOLERANCE:
            min_diff = diff
            closest_channel_num = channel_num
    return closest_channel_num

def get_dominant_frequency(data, rate):
    """Calculates the dominant frequency in a given audio data segment."""
    if len(data) == 0 or np.max(np.abs(data)) < 0.005:
        return None
    window = np.hanning(len(data))
    data = data * window
    N = len(data)
    yf = fft(data)
    xf = fftfreq(N, 1 / rate)
    positive_mask = xf > 0
    xf_positive = xf[positive_mask]
    yf_positive = np.abs(yf[positive_mask])

    if len(yf_positive) == 0: return None

    try:
        idx = np.argmax(yf_positive)
        return xf_positive[idx]
    except IndexError:
        return None

def reset_decoder_state_variables():
    """Resets all decoder state variables to their initial values."""
    global decoder_state, current_message_channels_log, raw_decoded_payload_bytes, training_sequence_index
    global current_tone_candidate_nominal_chan, current_tone_candidate_blocks, fsm_informed_of_this_segment
    global current_tone_candidate_freq_samples
    global current_high_nibble_value, byte_processing_state, current_byte_channels_debug
    global current_message_type, header_buffer, header_parsed, file_metadata, payload_bytes_received

    decoder_state = "IDLE"
    current_message_channels_log = []
    raw_decoded_payload_bytes = []
    training_sequence_index = 0
    current_tone_candidate_nominal_chan = None
    current_tone_candidate_blocks = 0
    fsm_informed_of_this_segment = False
    recent_detections_nominal_chan.clear()
    current_tone_candidate_freq_samples = []

    current_high_nibble_value = None
    byte_processing_state = "EXPECT_HIGH_NIBBLE"
    current_byte_channels_debug = []

    # File/Message specific variables reset
    current_message_type = None
    header_buffer = []
    header_parsed = False
    file_metadata = {
        "filename": "",
        "extension": "",
        "payload_size": 0 # Reset to 0
    }
    payload_bytes_received = 0


def reset_decoder_soft():
    """Soft reset: Clears message data but keeps calibration."""
    print("Decoder soft reset. Waiting for preamble...")
    reset_decoder_state_variables()

def reset_decoder_full_including_calibration():
    """Full reset: Clears everything including calibration data."""
    global calibrated_frequencies, has_been_calibrated
    print("Decoder FULL reset (calibration lost). Waiting for preamble...")
    calibrated_frequencies = {}
    has_been_calibrated = False
    reset_decoder_state_variables()

def print_progress_bar(current, total, bar_length=40):
    if total == 0: # Avoid division by zero
        percent = 0
    else:
        percent = float(current) / total
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f"\rReceiving: [{arrow + spaces}] {int(percent * 100)}% ({current}/{total} bytes)")
    sys.stdout.flush()

def reset_decoder_after_message_or_error():
    """Called after a message is completed/interrupted or an error occurs."""
    global raw_decoded_payload_bytes, current_message_channels_log, current_message_type
    global file_metadata, payload_bytes_received

    # Clear any live message/progress bar from the console
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()

    print(f"\n--- MESSAGE END / DECODER RESET ---")
    if decoder_state == "CALIBRATING" and not has_been_calibrated:
         print(f"Status: Calibration failed or interrupted.")
    elif not raw_decoded_payload_bytes and (decoder_state == "RECEIVING_DATA" or decoder_state == "READING_HEADER" or (decoder_state == "IDLE" and current_message_channels_log)):
        print(f"Status: Message/File decoding failed or interrupted (no data received or partial header).")
    elif raw_decoded_payload_bytes:
        # The raw_decoded_payload_bytes contains header + actual data payload.
        # We need to extract only the actual data payload part.
        
        raw_data_payload = b''
        if header_parsed:
            # The start of the data payload is current length of raw_decoded_payload_bytes - payload_bytes_received
            data_start_idx = len(raw_decoded_payload_bytes) - payload_bytes_received
            if data_start_idx >= 0:
                raw_data_payload = bytes(raw_decoded_payload_bytes[data_start_idx:])
            else:
                print("Warning: Could not extract raw data payload due to index error. Raw payload might be corrupted.")
        else:
            print("Warning: Cannot process data, header was not fully parsed.")

        if current_message_type == MESSAGE_TYPE_FILE:
            file_name = file_metadata["filename"]
            file_ext = file_metadata["extension"]
            full_path = f"{file_name}.{file_ext}" if file_ext else file_name
            
            try:
                with open(full_path, 'wb') as f:
                    f.write(raw_data_payload)
                print(f"Status: File '{full_path}' successfully received and saved ({len(raw_data_payload)} bytes).")
            except IOError as e:
                print(f"Status: File '{full_path}' received, but failed to save: {e}")
            
            print(f"Metadata - Payload Size: {file_metadata['payload_size']} bytes")
            print(f"  Filename: '{file_name}', Extension: '{file_ext}'")
        elif current_message_type == MESSAGE_TYPE_TEXT:
            try:
                decoded_text = raw_data_payload.decode('utf-8')
                print(f"Status: Text message decoded successfully.")
                print(f"Decoded Message: '{decoded_text}'")
            except UnicodeDecodeError:
                print(f"Status: Text message received, but could not be decoded as UTF-8.")
                print(f"Raw Bytes (first 50): {raw_data_payload[:50]}")
        else: # Should not happen if message_type is set
            print(f"Status: Unknown message type detected ({current_message_type}). Raw payload received.")
            
    else:
        print(f"Status: Reset triggered without significant message activity.")

    print(f"Raw Confirmed Channels (Detected IDs): {current_message_channels_log}")
    print(f"Raw Assembled Payload Bytes (Hex, first 50): {[f'{b:02X}' for b in raw_decoded_payload_bytes[:50]]}...")

    if has_been_calibrated and calibrated_frequencies:
        print("\nCurrent Calibrated Frequencies (Hz):")
        for ch_num in sorted(calibrated_frequencies.keys()):
            if ch_num in NOMINAL_CHANNEL_FREQUENCIES:
                print(f"  Ch {ch_num:2d}: {calibrated_frequencies[ch_num]:.1f} (Nominal: {NOMINAL_CHANNEL_FREQUENCIES[ch_num]:.1f})")
    print("-----------------------------------\n")

    reset_decoder_state_variables()

def process_decoded_byte(byte_value):
    """
    Handles appending a newly assembled byte to either the header_buffer or
    the raw_decoded_payload_bytes, and triggers header parsing or progress updates.
    """
    global header_parsed, raw_decoded_payload_bytes, payload_bytes_received
    global decoder_state # Need to update state if header parsing completes

    if not header_parsed:
        header_buffer.append(byte_value)
        if parse_header(): # parse_header also updates header_parsed and decoder_state to RECEIVING_DATA
            # Header is now parsed. Copy header_buffer contents to payload.
            raw_decoded_payload_bytes.extend(header_buffer) 
            header_buffer.clear() # Clear buffer as it's been processed
    else: # Header is already parsed, so we are receiving data payload
        raw_decoded_payload_bytes.append(byte_value)
        payload_bytes_received += 1
        print_progress_bar(payload_bytes_received, file_metadata["payload_size"])
        # Check if all payload bytes are received
        if payload_bytes_received >= file_metadata["payload_size"] and file_metadata["payload_size"] > 0:
            print(f"\nAll payload data ({payload_bytes_received} bytes) received. Waiting for Postamble...")
            # The decoder state remains "RECEIVING_DATA" until postamble or error
            # We do NOT reset here. The postamble triggers final reset and processing.


def parse_header():
    """
    Attempts to parse the header from header_buffer.
    Returns True if header is complete and valid, False otherwise.
    Sets global file_metadata and current_message_type.
    """
    global header_parsed, current_message_type, file_metadata, decoder_state

    # Minimum header size: START (1) + TYPE (1) + PAYLOAD_SIZE (4) + END (1) = 7 bytes for text
    # Minimum fixed part for file is more complex due to dynamic length fields.

    if len(header_buffer) < 2: # Need at least start delimiter and message type
        return False

    if header_buffer[0] != HEADER_START_DELIMITER:
        print(f"Error: Header does not start with delimiter {HEADER_START_DELIMITER:02X}. Resetting.")
        reset_decoder_after_message_or_error()
        return False

    current_message_type = header_buffer[1]

    if current_message_type == MESSAGE_TYPE_TEXT:
        # Text header: FE (1) + 00 (1) + DATA_SIZE (4) + FF (1) = 7 bytes
        if len(header_buffer) < 7:
            return False # Not enough bytes for full text header yet

        if header_buffer[6] != HEADER_END_DELIMITER:
            print(f"Error: Text header does not end with delimiter {HEADER_END_DELIMITER:02X}. Resetting.")
            reset_decoder_after_message_or_error()
            return False
        
        file_metadata["payload_size"] = bytes_to_int_le(header_buffer[2:6])
        header_parsed = True
        print(f"Text Message Header Parsed. Raw Data Size: {file_metadata['payload_size']} bytes. Waiting for raw text data.")
        decoder_state = "RECEIVING_DATA" # Transition to receiving data payload
        return True

    elif current_message_type == MESSAGE_TYPE_FILE:
        # File header: FE (1) + 01 (1) + FNL (2) + FN (N) + EL (2) + EXT (N) + FS (4) + FF (1)
        # Minimum fixed part: Start, Type, FNL, EL, FS, End = 1+1+2+2+4+1 = 11 bytes
        fixed_file_header_min_len = 11 
        if len(header_buffer) < fixed_file_header_min_len:
            return False # Not enough bytes for fixed part of file header yet

        # Extract lengths and sizes to determine full header length
        try:
            filename_len = bytes_to_int_le(header_buffer[2:4])
            
            # Calculate where extension length bytes *should* start
            ext_len_start_idx = 4 + filename_len
            if len(header_buffer) < ext_len_start_idx + 2: return False # Not enough for ext length yet

            extension_len = bytes_to_int_le(header_buffer[ext_len_start_idx : ext_len_start_idx + 2])

            # Calculate where file size bytes *should* start
            file_size_start_idx = ext_len_start_idx + 2 + extension_len
            if len(header_buffer) < file_size_start_idx + 4: return False # Not enough for file size yet

            file_size = bytes_to_int_le(header_buffer[file_size_start_idx : file_size_start_idx + 4])
            
            # Calculate where the HEADER_END_DELIMITER *should* be
            end_delimiter_idx = file_size_start_idx + 4

            if len(header_buffer) < end_delimiter_idx + 1: return False # Need end delimiter

            if header_buffer[end_delimiter_idx] != HEADER_END_DELIMITER:
                print(f"Error: File header does not end with delimiter {HEADER_END_DELIMITER:02X}. Resetting.")
                reset_decoder_after_message_or_error()
                return False

            # All parts present, now extract strings
            filename_bytes = bytes(header_buffer[4 : 4 + filename_len])
            extension_bytes = bytes(header_buffer[ext_len_start_idx + 2 : ext_len_start_idx + 2 + extension_len])
            
            file_metadata["filename"] = filename_bytes.decode('utf-8')
            file_metadata["extension"] = extension_bytes.decode('utf-8')
            file_metadata["payload_size"] = file_size # This is the RAW file size
            
            header_parsed = True
            print(f"File Header Parsed. Filename: '{file_metadata['filename']}.{file_metadata['extension']}', "
                  f"Payload Size: {file_metadata['payload_size']} bytes. Waiting for raw file data.")
            decoder_state = "RECEIVING_DATA" # Transition to receiving data payload
            return True

        except (IndexError, UnicodeDecodeError, ValueError) as e:
            print(f"Error parsing file header: {e}. Header buffer: {header_buffer}. Resetting.")
            reset_decoder_after_message_or_error()
            return False
    else:
        print(f"Error: Unknown message type {current_message_type:02X}. Resetting.")
        reset_decoder_after_message_or_error()
        return False
        

def fsm_process_confirmed_tone(confirmed_channel_id_used_for_detection, actual_average_frequency):
    """
    State machine for processing confirmed tones.
    Manages calibration, message decoding, and state transitions.
    """
    global decoder_state, training_sequence_index, raw_decoded_payload_bytes
    global current_message_channels_log, calibrated_frequencies, has_been_calibrated
    global current_high_nibble_value, byte_processing_state, current_byte_channels_debug
    global header_buffer, header_parsed, payload_bytes_received
    # Add these globals for the added reset after calibration
    global current_tone_candidate_nominal_chan, current_tone_candidate_blocks, fsm_informed_of_this_segment, current_tone_candidate_freq_samples, recent_detections_nominal_chan


    if decoder_state == "IDLE":
        if confirmed_channel_id_used_for_detection == 1:
            print(f"Preamble Confirmed (Nominal Ch 1 @ {actual_average_frequency:.1f} Hz).")
            current_message_channels_log.append(1)
            decoder_state = "CALIBRATING"
            training_sequence_index = 0
            calibrated_frequencies = {} # Clear previous calibration
            has_been_calibrated = False
            calibrated_frequencies[1] = actual_average_frequency # Calibrate Preamble channel
            print(f"  Calibrated Ch 1 to {actual_average_frequency:.1f} Hz")
            print(f"State Transition: IDLE -> CALIBRATING. Waiting for Training Ch {TRAINING_SEQUENCE[0]}...")
        else:
            pass # Suppress repeated "Ignored Ch..." messages in IDLE state

    elif decoder_state == "CALIBRATING":
        if training_sequence_index < len(TRAINING_SEQUENCE):
            expected_calib_channel = TRAINING_SEQUENCE[training_sequence_index]
            nominal_freq_of_expected = NOMINAL_CHANNEL_FREQUENCIES.get(expected_calib_channel)

            if nominal_freq_of_expected is None:
                print(f"Error: Expected calibration channel {expected_calib_channel} not in nominal map. Resetting.")
                reset_decoder_full_including_calibration()
                return

            if abs(actual_average_frequency - nominal_freq_of_expected) <= FREQUENCY_TOLERANCE:
                sane_diff = abs(actual_average_frequency - nominal_freq_of_expected)
                sane_tolerance_hz = nominal_freq_of_expected * CALIBRATION_SANE_TOLERANCE_FACTOR

                if sane_diff <= sane_tolerance_hz:
                    calibrated_frequencies[expected_calib_channel] = actual_average_frequency
                    current_message_channels_log.append(expected_calib_channel) # Log the *expected* channel ID
                    print(f"  Calibrated Ch {expected_calib_channel} to {actual_average_frequency:.1f} Hz (Nominal: {nominal_freq_of_expected:.1f} Hz).")
                    training_sequence_index += 1

                    if training_sequence_index == len(TRAINING_SEQUENCE):
                        if len(calibrated_frequencies) == len(NOMINAL_CHANNEL_FREQUENCIES):
                            print("Calibration sequence complete.")
                            decoder_state = "READING_HEADER" # New state for header
                            has_been_calibrated = True
                            print("State Transition: CALIBRATING -> READING_HEADER. Waiting for message header...")
                            
                            # --- CRITICAL ADDITION: Reset tone tracking after calibration ---
                            # This ensures the first header tone is correctly detected as a new segment.
                            current_tone_candidate_nominal_chan = None
                            current_tone_candidate_blocks = 0
                            fsm_informed_of_this_segment = False
                            current_tone_candidate_freq_samples = []
                            recent_detections_nominal_chan.clear() # Clear deque too!
                            # --- END CRITICAL ADDITION ---

                        else:
                            print("Warning: Calibration sequence finished but not all channels recorded. Resetting.")
                            reset_decoder_full_including_calibration()
                    else:
                        print(f"  Waiting for Training Ch {TRAINING_SEQUENCE[training_sequence_index]}...")
                else:
                    print(f"Error: Calibration sanity check failed for Ch {expected_calib_channel}. Freq {actual_average_frequency:.1f} Hz is too far from nominal {nominal_freq_of_expected:.1f} Hz (>{sane_tolerance_hz:.1f} Hz). Resetting.")
                    reset_decoder_full_including_calibration()
            else:
                print(f"Error: During calibration for Ch {expected_calib_channel}, detected frequency {actual_average_frequency:.1f} Hz is outside {FREQUENCY_TOLERANCE} Hz tolerance from its nominal {nominal_freq_of_expected:.1f} Hz. Resetting.")
                reset_decoder_full_including_calibration()
        else:
            print("Error: training_sequence_index out of bounds in CALIBRATING. Resetting.")
            reset_decoder_full_including_calibration()

    elif decoder_state == "READING_HEADER" or decoder_state == "RECEIVING_DATA":
        data_channel = confirmed_channel_id_used_for_detection

        if data_channel == 1: # Postamble
            print(f"Postamble Confirmed (Calibrated Ch 1 @ {actual_average_frequency:.1f} Hz).")
            # If we received a postamble, it means message is done. Handle incomplete states.
            if header_parsed:
                if payload_bytes_received < file_metadata["payload_size"]:
                    print(f"  Warning: Payload transmission ended prematurely. Expected {file_metadata['payload_size']} bytes, got {payload_bytes_received}.")
            else: # Header was not parsed completely
                print(f"  Warning: Message ended without a complete header. Current header_buffer: {[f'{b:02X}' for b in header_buffer]}.")

            current_message_channels_log.append(1)
            reset_decoder_after_message_or_error()
            return

        current_message_channels_log.append(data_channel)

        if 4 <= data_channel <= 19: # Hex digit (0-F)
            hex_val = data_channel - 4
            current_byte_channels_debug.append(data_channel)

            if byte_processing_state == "EXPECT_HIGH_NIBBLE":
                current_high_nibble_value = hex_val
                byte_processing_state = "EXPECT_LOW_NIBBLE"
            elif byte_processing_state == "EXPECT_LOW_NIBBLE":
                if current_high_nibble_value is not None:
                    full_byte_val = (current_high_nibble_value << 4) | hex_val
                    process_decoded_byte(full_byte_val) # Call helper here
                    current_high_nibble_value = None
                    byte_processing_state = "EXPECT_BYTE_SEPARATOR"
                    current_byte_channels_debug = []
                else:
                    print(f"Error: Low Nibble (Ch {data_channel}) received without a registered High Nibble. Resetting.")
                    reset_decoder_after_message_or_error()
            else:
                print(f"Error: Hex digit Ch {data_channel} received at unexpected state ({byte_processing_state}). Resetting.")
                reset_decoder_after_message_or_error()

        elif data_channel == 2: # Byte Separator
            current_byte_channels_debug.append(data_channel)
            if byte_processing_state == "EXPECT_BYTE_SEPARATOR":
                byte_processing_state = "EXPECT_HIGH_NIBBLE"
            elif byte_processing_state == "EXPECT_LOW_NIBBLE" and current_high_nibble_value is not None:
                # --- ERROR RECOVERY (Duplicate Nibble) - Warning suppressed ---
                # print(f"Warning: Missing Low Nibble for byte (expected Ch 4-19, got Ch 2). Assuming duplicate of High Nibble ({current_high_nibble_value:X}).")
                full_byte_val = (current_high_nibble_value << 4) | current_high_nibble_value
                process_decoded_byte(full_byte_val) # Process the reconstructed byte
                
                current_high_nibble_value = None # Clear for next byte
                byte_processing_state = "EXPECT_HIGH_NIBBLE" # Ready for next byte's high nibble
                current_byte_channels_debug = [] # Clear debug for this "repaired" byte
            else:
                print(f"Error: Byte Separator (Ch 2) received at unexpected state ({byte_processing_state}). Resetting.")
                reset_decoder_after_message_or_error()

        elif data_channel == 3: # Ch3 is only part of calibration/training now. If seen during message, it's an error.
             print(f"Error: Ch 3 detected in message phase. This channel is not used as a separator anymore. Resetting.")
             reset_decoder_after_message_or_error()

        else: # Unrecognized channel
            print(f"Warning: Unexpected Ch {data_channel} ({actual_average_frequency:.1f} Hz) in message state. Resetting message attempt.")
            reset_decoder_after_message_or_error()
    else:
        print(f"Warning: FSM in unhandled state: {decoder_state}. Full reset.")
        reset_decoder_full_including_calibration()

def audio_callback(indata, frames, time_info, status):
    """
    Audio stream callback function. Processes incoming audio blocks.
    Identifies dominant frequencies and feeds them to the FSM.
    """
    global current_tone_candidate_nominal_chan, current_tone_candidate_blocks, fsm_informed_of_this_segment
    global decoder_state, current_tone_candidate_freq_samples

    if status:
        pass # Suppress "Input overflow" warnings if they happen too frequently and are harmless

    mono_data = indata[:, 0] if indata.ndim > 1 else indata
    dominant_freq_this_block = get_dominant_frequency(mono_data, SAMPLING_RATE)

    use_nominal_map_for_this_detection = (decoder_state == "IDLE" or decoder_state == "CALIBRATING")
    detected_channel_this_block = find_closest_channel(dominant_freq_this_block, use_nominal_map_only=use_nominal_map_for_this_detection)

    recent_detections_nominal_chan.append(detected_channel_this_block)

    if len(recent_detections_nominal_chan) < recent_detections_nominal_chan.maxlen:
        return

    counts = collections.Counter(d for d in recent_detections_nominal_chan if d is not None)
    stable_detected_channel_candidate = None

    if counts:
        most_common_ch, num_most_common = counts.most_common(1)[0]
        if num_most_common >= recent_detections_nominal_chan.maxlen -1 :
            stable_detected_channel_candidate = most_common_ch

    if stable_detected_channel_candidate == current_tone_candidate_nominal_chan:
        if current_tone_candidate_nominal_chan is not None:
            current_tone_candidate_blocks += 1
            if dominant_freq_this_block is not None:
                 current_tone_candidate_freq_samples.append(dominant_freq_this_block)
    else:
        current_tone_candidate_nominal_chan = stable_detected_channel_candidate
        current_tone_candidate_blocks = 1 if stable_detected_channel_candidate is not None else 0
        fsm_informed_of_this_segment = False
        current_tone_candidate_freq_samples = []
        if dominant_freq_this_block is not None and stable_detected_channel_candidate is not None:
            current_tone_candidate_freq_samples.append(dominant_freq_this_block)

    if current_tone_candidate_nominal_chan is not None and not fsm_informed_of_this_segment:
        is_long_duration_type_context = (current_tone_candidate_nominal_chan == 1)
        min_blocks_needed = MIN_CONSECUTIVE_BLOCKS_FOR_PREAMBLE if is_long_duration_type_context else MIN_CONSECUTIVE_BLOCKS_FOR_CHANNEL

        if current_tone_candidate_blocks >= min_blocks_needed:
            avg_freq_for_segment = None
            if current_tone_candidate_freq_samples:
                avg_freq_for_segment = np.mean(current_tone_candidate_freq_samples)
            
            if avg_freq_for_segment is None and current_tone_candidate_nominal_chan is not None:
                 current_map = get_channel_map_for_find()
                 fallback_freq = current_map.get(current_tone_candidate_nominal_chan, NOMINAL_CHANNEL_FREQUENCIES.get(current_tone_candidate_nominal_chan, 0))
                 if fallback_freq != 0:
                     avg_freq_for_segment = fallback_freq

            if avg_freq_for_segment is not None:
                fsm_process_confirmed_tone(current_tone_candidate_nominal_chan, avg_freq_for_segment)
                fsm_informed_of_this_segment = True


# --- Main Program ---
if __name__ == "__main__":
    print("Available audio input devices:")
    print(sd.query_devices())

    INPUT_DEVICE_ID = None # Set to an integer device ID if you have multiple and want to select one

    print(f"\nListening for BPFSK signals (Hexadecimal Protocol, 10-18kHz range, Raw Files/Text)...")
    print(f"Nominal Channel Frequencies (Hz, linearly spaced):")
    for ch_num in sorted(NOMINAL_CHANNEL_FREQUENCIES.keys()):
        print(f"  Ch {ch_num:2d}: {NOMINAL_CHANNEL_FREQUENCIES[ch_num]:.1f} Hz")
    
    print(f"\nTarget Channel Duration: {CHANNEL_DURATION*1000:.0f} ms")
    print(f"Preamble Duration: {PREAMBLE_DURATION*1000:.0f} ms")
    print(f"Analysis Blocksize: {BLOCKSIZE_SECONDS*1000:.0f} ms ({BLOCKSIZE_SAMPLES} samples)")
    print(f"FFT Freq. Resolution: ~{SAMPLING_RATE/BLOCKSIZE_SAMPLES:.0f} Hz/bin")
    print(f"Min blocks for channel tone: {MIN_CONSECUTIVE_BLOCKS_FOR_CHANNEL} ({MIN_CONSECUTIVE_BLOCKS_FOR_CHANNEL*BLOCKSIZE_SECONDS*1000:.0f} ms)")
    print(f"Min blocks for preamble: {MIN_CONSECUTIVE_BLOCKS_FOR_PREAMBLE} ({MIN_CONSECUTIVE_BLOCKS_FOR_PREAMBLE*BLOCKSIZE_SECONDS*1000:.0f} ms)")
    print(f"Frequency Tolerance: +/- {FREQUENCY_TOLERANCE} Hz")
    print(f"Calibration Sanity Tolerance: +/- {CALIBRATION_SANE_TOLERANCE_FACTOR*100:.1f}% of nominal frequency.")
    print(f"Received files will be saved in: {os.getcwd()}")


    reset_decoder_soft()
    try:
        with sd.InputStream(device=INPUT_DEVICE_ID, channels=1, samplerate=SAMPLING_RATE,
                            blocksize=BLOCKSIZE_SAMPLES, callback=audio_callback):
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping decoder.")
        reset_decoder_after_message_or_error()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
