import sounddevice as sd
import numpy as np
from scipy.fft import fft, fftfreq
import time
import collections

# --- Configuration ---
SAMPLING_RATE = 44100
CHANNEL_DURATION = 0.03
PREAMBLE_DURATION = 1.0
FREQUENCY_TOLERANCE = 50

# In your Python script
NOMINAL_CHANNEL_FREQUENCIES = {
    1: 5000,
    2: 6000,
    3: 7000,
    4: 8000,
    5: 9000,
    6: 10000,
    7: 11000,
    8: 12000,
    9: 13000,
    10: 14000
}

BLOCKSIZE_SECONDS = 0.01
BLOCKSIZE_SAMPLES = int(SAMPLING_RATE * BLOCKSIZE_SECONDS)

DETECTION_THRESHOLD_FACTOR_CHANNEL = 0.51
DETECTION_THRESHOLD_FACTOR_PREAMBLE = 0.7

MIN_CONSECUTIVE_BLOCKS_FOR_CHANNEL = max(1, int(np.ceil((CHANNEL_DURATION / BLOCKSIZE_SECONDS) * DETECTION_THRESHOLD_FACTOR_CHANNEL)))
MIN_CONSECUTIVE_BLOCKS_FOR_PREAMBLE = max(1, int(np.ceil((PREAMBLE_DURATION / BLOCKSIZE_SECONDS) * DETECTION_THRESHOLD_FACTOR_PREAMBLE)))

TRAINING_SEQUENCE = [2, 3, 4, 5, 6, 7, 8, 9, 10]
CALIBRATION_SANE_TOLERANCE_FACTOR = 0.075

decoder_state = "IDLE"
current_message_channels_log = []
decoded_message_chars = []
current_byte_value = 0
training_sequence_index = 0
calibrated_frequencies = {}
has_been_calibrated = False
current_tone_candidate_nominal_chan = None
current_tone_candidate_blocks = 0
fsm_informed_of_this_segment = False
recent_detections_nominal_chan = collections.deque(maxlen=3)
current_tone_candidate_freq_samples = []

def get_channel_map_for_find():
    if has_been_calibrated and calibrated_frequencies:
        return calibrated_frequencies
    return NOMINAL_CHANNEL_FREQUENCIES

def find_closest_channel(frequency, use_nominal_map_only=False):
    if frequency is None: return None
    source_freq_map = NOMINAL_CHANNEL_FREQUENCIES if use_nominal_map_only else get_channel_map_for_find()
    if not source_freq_map or (has_been_calibrated and len(source_freq_map) < len(NOMINAL_CHANNEL_FREQUENCIES)-1):
        source_freq_map = NOMINAL_CHANNEL_FREQUENCIES
    min_diff = float('inf')
    closest_channel_num = None
    for channel_num, freq_val in source_freq_map.items():
        diff = abs(frequency - freq_val)
        local_tolerance = FREQUENCY_TOLERANCE
        if diff < min_diff and diff < local_tolerance:
            min_diff = diff
            closest_channel_num = channel_num
    return closest_channel_num

def get_dominant_frequency(data, rate):
    if len(data) == 0 or np.max(np.abs(data)) < 0.005: return None
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
    except IndexError: return None

def reset_decoder_state_variables():
    global decoder_state, current_message_channels_log, decoded_message_chars, current_byte_value, training_sequence_index
    global current_tone_candidate_nominal_chan, current_tone_candidate_blocks, fsm_informed_of_this_segment
    global current_tone_candidate_freq_samples

    decoder_state = "IDLE"
    current_message_channels_log = []
    decoded_message_chars = []
    current_byte_value = 0
    training_sequence_index = 0
    current_tone_candidate_nominal_chan = None
    current_tone_candidate_blocks = 0
    fsm_informed_of_this_segment = False
    recent_detections_nominal_chan.clear()
    current_tone_candidate_freq_samples = []

def reset_decoder_soft():
    print("Decoder soft reset. Waiting for preamble...")
    reset_decoder_state_variables()

def reset_decoder_full_including_calibration():
    global calibrated_frequencies, has_been_calibrated
    print("Decoder FULL reset (calibration lost). Waiting for preamble...")
    calibrated_frequencies = {}
    has_been_calibrated = False
    reset_decoder_state_variables()

def reset_decoder_after_message_or_error():
    global decoded_message_chars, current_message_channels_log
    if decoded_message_chars or current_message_channels_log:
        full_message = "".join(c for c in decoded_message_chars if c != '(invalid_ascii)')
        print(f"\n--- MESSAGE END / DECODER RESET ---")
        if decoder_state == "CALIBRATING" and not has_been_calibrated:
             print(f"Status: Calibration failed or interrupted.")
        elif not decoded_message_chars and (decoder_state == "IN_MESSAGE" or (decoder_state == "IDLE" and current_message_channels_log)): # Error during message
            print(f"Status: Message decoding failed or interrupted.")
        print(f"Decoded: {full_message if full_message else '(empty)'}")
        print(f"Raw Confirmed Channels (Detected IDs): {current_message_channels_log}")
        if has_been_calibrated and calibrated_frequencies:
            print("Current Calibrated Frequencies (Hz):")
            if 1 in calibrated_frequencies:
                 print(f"  Ch 1: {calibrated_frequencies[1]:.1f} (Nominal: {NOMINAL_CHANNEL_FREQUENCIES.get(1, 'N/A')})")
            for ch_num in TRAINING_SEQUENCE:
                if ch_num in calibrated_frequencies:
                    print(f"  Ch {ch_num}: {calibrated_frequencies[ch_num]:.1f} (Nominal: {NOMINAL_CHANNEL_FREQUENCIES.get(ch_num, 'N/A')})")
        print("-----------------------------------\n")
    else:
        print("Decoder reset (no significant activity). Waiting for preamble...")
    reset_decoder_state_variables()

def fsm_process_confirmed_tone(confirmed_channel_id_used_for_detection, actual_average_frequency):
    global decoder_state, current_byte_value, training_sequence_index, decoded_message_chars
    global current_message_channels_log, calibrated_frequencies, has_been_calibrated
    
    if decoder_state == "IDLE":
        if confirmed_channel_id_used_for_detection == 1: 
            print(f"Preamble Confirmed (Nominal Ch 1 @ {actual_average_frequency:.1f} Hz).")
            current_message_channels_log.append(1)
            decoder_state = "CALIBRATING"
            training_sequence_index = 0
            calibrated_frequencies = {} 
            has_been_calibrated = False
            calibrated_frequencies[1] = actual_average_frequency 
            print(f"  Calibrated Ch 1 to {actual_average_frequency:.1f} Hz")
            print(f"State Transition: IDLE -> CALIBRATING. Waiting for Training Ch {TRAINING_SEQUENCE[0]}...")
        else: # Ignore other channels if IDLE
             print(f"Info: Ignored Ch {confirmed_channel_id_used_for_detection} ({actual_average_frequency:.1f} Hz) in IDLE state (expected Preamble Ch 1).")


    elif decoder_state == "CALIBRATING":
        if training_sequence_index < len(TRAINING_SEQUENCE):
            expected_calib_channel = TRAINING_SEQUENCE[training_sequence_index]
            nominal_freq_of_expected = NOMINAL_CHANNEL_FREQUENCIES[expected_calib_channel]
            nominally_identified_channel_for_this_freq = find_closest_channel(actual_average_frequency, use_nominal_map_only=True)

            if nominally_identified_channel_for_this_freq == expected_calib_channel:
                sane_diff = abs(actual_average_frequency - nominal_freq_of_expected)
                sane_tolerance_hz = nominal_freq_of_expected * CALIBRATION_SANE_TOLERANCE_FACTOR
                if sane_diff <= sane_tolerance_hz:
                    calibrated_frequencies[expected_calib_channel] = actual_average_frequency
                    current_message_channels_log.append(expected_calib_channel)
                    print(f"  Calibrated Ch {expected_calib_channel} to {actual_average_frequency:.1f} Hz (Nominal: {nominal_freq_of_expected}).")
                    training_sequence_index += 1
                    if training_sequence_index == len(TRAINING_SEQUENCE):
                        print("Calibration sequence complete.")
                        has_been_calibrated = True
                        decoder_state = "IN_MESSAGE"
                        current_byte_value = 0 
                        fsm_process_confirmed_tone.current_byte_channels_debug = [] 
                        print("State Transition: CALIBRATING -> IN_MESSAGE. Waiting for data...")
                    else:
                        print(f"  Waiting for Training Ch {TRAINING_SEQUENCE[training_sequence_index]}...")
                else: 
                    print(f"Error: Calibration sanity check failed for Ch {expected_calib_channel}. Freq {actual_average_frequency:.1f} Hz is too far from nominal {nominal_freq_of_expected} Hz. Resetting.")
                    reset_decoder_full_including_calibration()
            else: 
                print(f"Error: During calibration for Ch {expected_calib_channel}, freq {actual_average_frequency:.1f} Hz maps to Ch {nominally_identified_channel_for_this_freq}. Resetting.")
                reset_decoder_full_including_calibration()
        else: 
            print("Error: training_sequence_index out of bounds in CALIBRATING. Resetting.")
            reset_decoder_full_including_calibration()

    elif decoder_state == "IN_MESSAGE":
        if not hasattr(fsm_process_confirmed_tone, 'current_byte_channels_debug'):
            fsm_process_confirmed_tone.current_byte_channels_debug = []
        data_channel = confirmed_channel_id_used_for_detection
        if data_channel == 1: 
            print(f"Postamble Confirmed (Calibrated Ch 1 @ {actual_average_frequency:.1f} Hz).")
            if current_byte_value > 0 or fsm_process_confirmed_tone.current_byte_channels_debug:
                char_str = ""
                try: char_str = chr(current_byte_value)
                except: char_str = "(invalid_ascii)"
                print(f"  Partial byte at end: Value={current_byte_value} (0b{current_byte_value:08b}), Channels={fsm_process_confirmed_tone.current_byte_channels_debug} -> '{char_str}'")
                if char_str != "(invalid_ascii)": decoded_message_chars.append(char_str)
            current_message_channels_log.append(1)
            reset_decoder_after_message_or_error()
            return
        current_message_channels_log.append(data_channel)
        if 3 <= data_channel <= 10: 
            bit_position = data_channel - 3
            current_byte_value |= (1 << bit_position)
            fsm_process_confirmed_tone.current_byte_channels_debug.append(data_channel)
        elif data_channel == 2: 
            char_str = ""
            if current_byte_value > 0 or fsm_process_confirmed_tone.current_byte_channels_debug:
                try: char_str = chr(current_byte_value)
                except ValueError: char_str = "(invalid_ascii)"
                print(f"  Byte sep (Ch 2). Value={current_byte_value} (0b{current_byte_value:08b}), Channels={fsm_process_confirmed_tone.current_byte_channels_debug} -> Decoded '{char_str}'")
                if char_str != "(invalid_ascii)": decoded_message_chars.append(char_str)
                if decoded_message_chars:
                    valid_chars_so_far = [c for c in decoded_message_chars if c != '(invalid_ascii)']
                    if valid_chars_so_far: print(f"    Message so far: {''.join(valid_chars_so_far)}")
            else: 
                print(f"  Byte sep (Ch 2) received (empty byte: Value=0, Channels=[]).")
            current_byte_value = 0 
            fsm_process_confirmed_tone.current_byte_channels_debug = []
        elif data_channel is None:
            print("Info: Silence or unrecognized tone during message. Waiting...")
        else: 
            print(f"Warning: Unexpected Ch {data_channel} ({actual_average_frequency:.1f} Hz) in IN_MESSAGE state. Resetting message attempt.")
            reset_decoder_after_message_or_error()
    else:
        print(f"Warning: FSM in unhandled state: {decoder_state}. Full reset.")
        reset_decoder_full_including_calibration()

# Initialize the attribute on the function object after it's defined.
# This ensures it exists for the very first call to fsm_process_confirmed_tone if needed,
# though the primary initialization for message processing happens on state transition.
fsm_process_confirmed_tone.current_byte_channels_debug = []

def audio_callback(indata, frames, time_info, status):
    global current_tone_candidate_nominal_chan, current_tone_candidate_blocks, fsm_informed_of_this_segment
    global decoder_state, current_tone_candidate_freq_samples
    if status: print(status, flush=True)
    mono_data = indata[:, 0] if indata.ndim > 1 else indata
    dominant_freq_this_block = get_dominant_frequency(mono_data, SAMPLING_RATE)
    use_nominal_map_for_this_detection = (decoder_state == "IDLE" or decoder_state == "CALIBRATING")
    detected_channel_this_block = find_closest_channel(dominant_freq_this_block, use_nominal_map_only=use_nominal_map_for_this_detection)
    recent_detections_nominal_chan.append(detected_channel_this_block)
    if len(recent_detections_nominal_chan) < recent_detections_nominal_chan.maxlen: return
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
                 if fallback_freq != 0: avg_freq_for_segment = fallback_freq
            if avg_freq_for_segment is not None:
                fsm_process_confirmed_tone(current_tone_candidate_nominal_chan, avg_freq_for_segment)
                fsm_informed_of_this_segment = True

# --- Main Program ---
if __name__ == "__main__":
    print("Available audio input devices:")
    print(sd.query_devices())
    INPUT_DEVICE_ID = None 
    print(f"\nListening for BPFSK signals...")
    print(f"Target Channel Duration: {CHANNEL_DURATION*1000:.0f} ms")
    print(f"Analysis Blocksize: {BLOCKSIZE_SECONDS*1000:.0f} ms ({BLOCKSIZE_SAMPLES} samples)")
    print(f"FFT Freq. Resolution: ~{SAMPLING_RATE/BLOCKSIZE_SAMPLES:.0f} Hz/bin")
    print(f"Min blocks for channel tone: {MIN_CONSECUTIVE_BLOCKS_FOR_CHANNEL} ({MIN_CONSECUTIVE_BLOCKS_FOR_CHANNEL*BLOCKSIZE_SECONDS*1000:.0f} ms)")
    print(f"Min blocks for preamble: {MIN_CONSECUTIVE_BLOCKS_FOR_PREAMBLE} ({MIN_CONSECUTIVE_BLOCKS_FOR_PREAMBLE*BLOCKSIZE_SECONDS*1000:.0f} ms)")
    print(f"Frequency Tolerance: +/- {FREQUENCY_TOLERANCE} Hz")
    print(f"Calibration Sanity Tolerance: +/- {CALIBRATION_SANE_TOLERANCE_FACTOR*100:.1f}% of nominal frequency.")
    reset_decoder_soft()
    try:
        with sd.InputStream(device=INPUT_DEVICE_ID, channels=1, samplerate=SAMPLING_RATE,
                            blocksize=BLOCKSIZE_SAMPLES, callback=audio_callback):
            while True: time.sleep(0.5)
    except KeyboardInterrupt: print("\nStopping decoder."); reset_decoder_after_message_or_error() 
    except Exception as e: print(f"An error occurred: {e}"); import traceback; traceback.print_exc()
