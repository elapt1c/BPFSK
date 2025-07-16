""" This example repeatedly transmits text or file data using the HFSK protocol
    (Hexadecimal Frequency Shift Keying) by bit-banging a GPIO pin on a Raspberry Pi Pico.
    
    Connect `board.GP16` (or your chosen pin) to a suitable transducer or RF circuit.
    A small speaker with a series resistor (e.g., 220-470 Ohm) can be used for audible testing.
    
    This code assumes CircuitPython is installed on the Pico and the `rp2pio` library is available.
"""

import time
import board
import rp2pio
import adafruit_pioasm

# --- HFSK Protocol Configuration (MUST MATCH RECEIVER) ---
MIN_OPERATING_FREQ_HZ = 10000
MAX_OPERATING_FREQ_HZ = 18000
NUM_TOTAL_CHANNELS = 19 # Channels 1 through 19

# Calculate the frequency step for linear spacing
FREQ_STEP = (MAX_OPERATING_FREQ_HZ - MIN_OPERATING_FREQ_HZ) / (NUM_TOTAL_CHANNELS - 1)

CHANNEL_FREQUENCIES = {}
for i in range(1, NUM_TOTAL_CHANNELS + 1):
    CHANNEL_FREQUENCIES[i] = MIN_OPERATING_FREQ_HZ + (i - 1) * FREQ_STEP

# --- Timing Constants ---
# Duration of a single channel tone (Pico's 'DI' now maps to CHANNEL_DURATION)
CHANNEL_DURATION = 0.03  # 30 ms
PREAMBLE_DURATION = 1.0  # 1 second for preamble/postamble

# --- Protocol Constants (MUST MATCH RECEIVER) ---
HEADER_START_DELIMITER = 0xFE
MESSAGE_TYPE_TEXT = 0x00
MESSAGE_TYPE_FILE = 0x01
HEADER_END_DELIMITER = 0xFF

# --- PIO Program for Square Wave Generation ---
# This PIO program generates a square wave by rapidly toggling the pin.
# `set pins, 0` sets the pin low (1 cycle).
# `set pins, 1` sets the pin high (1 cycle).
# A full period is 2 PIO clock cycles.
# So, desired_output_frequency = PIO_SM_clock_frequency / 2.
# Therefore, PIO_SM_clock_frequency = desired_output_frequency * 2.
PIO_CYCLES_PER_PERIOD = 2 # Based on the pio_generator program below

# The PIO program to generate a square wave
# Note: No `[delay]` is used here, aiming for max precision.
pio_generator = adafruit_pioasm.assemble(
    """
    .program hfsk_tx
    set pins, 0 ; Set pin low
    set pins, 1 ; Set pin high
    """
)

# --- Helper Functions ---

def to_little_endian_bytes(value, num_bytes):
    """Converts an integer to a little-endian bytearray."""
    return value.to_bytes(num_bytes, 'little')

def transmit_channel(channel_id, duration):
    """
    Transmits a square wave signal at the specified HFSK channel ID for the given duration
    directly on a GPIO pin using PIO.
    """
    if channel_id not in CHANNEL_FREQUENCIES:
        print(f"Error: Invalid channel ID {channel_id}")
        return

    target_output_frequency = CHANNEL_FREQUENCIES[channel_id]
    
    # Calculate the required PIO StateMachine clock frequency to achieve the target_output_frequency
    # PIO_SM_clock_frequency = target_output_frequency * PIO_CYCLES_PER_PERIOD
    # The Pico's system clock is typically 125 MHz. Ensure sm_frequency doesn't exceed this.
    # For 10kHz-18kHz, this is well within limits.
    sm_frequency = int(target_output_frequency * PIO_CYCLES_PER_PERIOD)

    # Instantiate and start the PIO StateMachine
    sm = rp2pio.StateMachine(
        pio_generator,
        frequency=sm_frequency,
        first_set_pin=board.GP16, # The GPIO pin to output the square wave
        set_pins=(board.GP16,)    # Pins that `set pins` instruction affects
    )
    # print(f"Transmitting Ch {channel_id} ({target_output_frequency:.1f} Hz) for {duration*1000:.0f} ms...")
    time.sleep(duration)  # Keep the signal active for the specified duration
    sm.deinit()  # Stop the signal and release PIO resources

def create_header(message_type, raw_data_bytes, filename="", file_extension=""):
    """
    Creates the protocol header for text or file data.
    Returns a bytearray.
    """
    header_parts = bytearray()
    header_parts.append(HEADER_START_DELIMITER)
    header_parts.append(message_type)

    if message_type == MESSAGE_TYPE_FILE:
        filename_bytes = filename.encode('utf-8')
        extension_bytes = file_extension.encode('utf-8')

        header_parts.extend(to_little_endian_bytes(len(filename_bytes), 2)) # Filename Length (UInt16)
        header_parts.extend(filename_bytes)
        header_parts.extend(to_little_endian_bytes(len(extension_bytes), 2)) # Extension Length (UInt16)
        header_parts.extend(extension_bytes)
        header_parts.extend(to_little_endian_bytes(len(raw_data_bytes), 4)) # Raw File Data Size (UInt32)
    elif message_type == MESSAGE_TYPE_TEXT:
        header_parts.extend(to_little_endian_bytes(len(raw_data_bytes), 4)) # Raw Text Data Size (UInt32)
    
    header_parts.append(HEADER_END_DELIMITER)
    return header_parts

def get_channel_sequence(payload_bytes):
    """
    Generates the sequence of channel IDs for the full payload (header + data).
    """
    channel_sequence = []

    # Preamble
    channel_sequence.append(1)

    # Training Sequence: Channel 2, then 3, then channels 4-19
    training_sequence = [2, 3] 
    for i in range(4, 20): # Channels 4 through 19
        training_sequence.append(i)
    channel_sequence.extend(training_sequence)

    # Convert payload bytes to channel IDs
    for byte_val in payload_bytes:
        high_nibble = (byte_val >> 4) & 0xF  # Get the 4 most significant bits (0-15)
        low_nibble = byte_val & 0xF         # Get the 4 least significant bits (0-15)

        # Transmit High Nibble (channel 4-19 for values 0-F)
        # hex_value (0-15) + 4 = channel_id (4-19)
        channel_sequence.append(high_nibble + 4)

        # Transmit Low Nibble (channel 4-19 for values 0-F)
        channel_sequence.append(low_nibble + 4)

        # Transmit Byte Separator (Channel 2) after each completed byte
        channel_sequence.append(2)
    
    # Postamble
    channel_sequence.append(1)
    
    return channel_sequence

def transmit_full_message(message_type, raw_data, filename="", file_extension=""):
    """
    Builds the header and payload, then transmits the full channel sequence.
    """
    print(f"Preparing transmission of type: {'TEXT' if message_type == MESSAGE_TYPE_TEXT else 'FILE'}")
    print(f"Raw data size: {len(raw_data)} bytes")

    header_bytes = create_header(message_type, raw_data, filename, file_extension)
    print(f"Header size: {len(header_bytes)} bytes")

    # Combine header and raw data into the full payload
    full_payload = bytearray()
    full_payload.extend(header_bytes)
    full_payload.extend(raw_data)
    print(f"Total payload to encode: {len(full_payload)} bytes")

    channel_sequence = get_channel_sequence(full_payload)
    print(f"Total channel tones to transmit: {len(channel_sequence)}")

    print("\n--- Starting Transmission ---")
    
    for i, channel_id in enumerate(channel_sequence):
        duration = CHANNEL_DURATION
        # Check for Preamble (first channel 1) or Postamble (last channel 1)
        if channel_id == 1 and (i == 0 or i == len(channel_sequence) - 1):
            duration = PREAMBLE_DURATION
        
        transmit_channel(channel_id, duration)
        # Optional: Add a very small inter-tone silence if needed for stability
        # time.sleep(0.001) 

    print("--- Transmission Complete ---")


# --- Example Usage on Raspberry Pi Pico ---

# Example 1: Transmit a text message
text_message = "Hello HFSK! This is a test message from a Raspberry Pi Pico. HFSK protocol in action!"
text_bytes = text_message.encode('utf-8')
print("\n--- Example 1: Transmitting Text ---")
transmit_full_message(MESSAGE_TYPE_TEXT, text_bytes)
time.sleep(5) # Pause before next transmission

# Example 2: Transmit a dummy file (replace with actual file content if desired)
# To read a file from Pico's filesystem:
# try:
#     with open("test_file.txt", "rb") as f:
#         dummy_file_content = f.read()
# except OSError:
#     print("test_file.txt not found. Creating dummy content.")
#     dummy_file_content = b"This is a dummy file content.\nIt has multiple lines.\n1234567890ABCDEF\n" * 2
# For demonstration, we'll create some dummy bytes directly:
dummy_file_content = b"This is a dummy file content.\nIt has multiple lines.\n1234567890ABCDEF\n" * 5
file_name = "pico_file"
file_ext = "log"
print(f"\n--- Example 2: Transmitting Dummy File '{file_name}.{file_ext}' ---")
transmit_full_message(MESSAGE_TYPE_FILE, dummy_file_content, filename=file_name, file_extension=file_ext)
time.sleep(5)

print("\n--- All examples finished. Looping transmissions indefinitely ---")
while True:
    transmit_full_message(MESSAGE_TYPE_TEXT, b"Pico repeating text. " * 3)
    time.sleep(10)
