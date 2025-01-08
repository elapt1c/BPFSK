""" This example repeatedly transmits ascii data using the BPFSK protocol using the gpio pin on a raspberry pi pico. no external hardware is required. """



import time
import board
import rp2pio
import adafruit_pioasm

# Define base frequency and bandwidth
base_freq = 2097152 # Base frequency in Hz
bandwidth = 512      # Bandwidth in Hz

# good signal shows up at ~1.48 mhz

# Timing constants
DI = 0.25  # Duration of a single bit, a good speed is 0.1

# Set up PIO assembler to generate frequency bursts using GPIO pins
generator = adafruit_pioasm.assemble(
    """
    set pins, 0  ; Set pin to low (start state)
    set pins, 1  ; Set pin to high (output)
    """
)

# Function to convert a message to a binary representation
def message_to_binary(message):
    """Convert a string message into a binary string (ASCII encoding)."""
    binary_message = ''
    for c in message:
        # Convert each character to binary with 8 bits (pad with leading zeros)
        binary_message += '{:08b}'.format(ord(c))  # Use format to ensure 8-bit binary
    return binary_message

# Function to transmit a frequency signal
def transmit_frequency(frequency, duration):
    """Transmit a signal at the given frequency for the specified duration."""
    sm = rp2pio.StateMachine(
        generator, frequency=frequency, first_set_pin=board.GP16
    )
    time.sleep(duration)  # Signal duration
    sm.deinit()  # Stop the signal after the duration

# Function to transmit a message over RF frequencies
def transmit_message(message):
    """Transmit a message over RF using frequency channels."""
    binary_message = message_to_binary(message)
    
    # Transmit preamble (Channel 1)
    transmit_frequency(base_freq + (bandwidth * 10), 0.25)
    transmit_frequency(base_freq + (bandwidth * 0), 0.25)
    transmit_frequency(base_freq + (bandwidth * 10), 0.25)
    transmit_frequency(base_freq + (bandwidth * 0), 0.25)
    # Start the transmission
    for byte_index in range(0, len(binary_message), 8):
        byte = binary_message[byte_index: byte_index + 8]
        
        # Transmit each bit of the byte
        for bit_index in range(8):
            if byte[bit_index] == '1':
                # Transmit the frequency for the corresponding bit position (using base_freq + offset)
                transmit_frequency(base_freq + (bit_index + 2) * bandwidth, DI)  # Channel 3-10
        
        # Transmit byte separator (Channel 2)
        transmit_frequency(base_freq + bandwidth, DI)
    
    transmit_frequency(base_freq + (bandwidth * 0), 1)
    
# Example usage
message = "This is a sample text for testing frequency-based data encoding. Each character is turned into binary, and read out, assigning each bit its own channel, then when the byte is finished a byte seperator channel 2 is transmitted. BPFSK protocol by elapt1c"
while True:
    transmit_message(message)
    time.sleep(5)


