
# HFSK-16
**H**ex **F**requency **S**hift **K**eying, Base **16**

## HOW IT WORKS:

This system transmits digital data (text or files) over an audio channel using Frequency Shift Keying (FSK). Each piece of data is encoded as a specific audio frequency (channel).

### Channels and Encoding:
*   There are **19 distinct channels** (frequencies) used in total, evenly spaced across the operating frequency range (e.g., 10kHz to 18kHz).
*   **Channel 1 (Preamble/Postamble):** Used at the very beginning and end of a transmission.
*   **Channel 2 (Byte Separator):** Signals the completion of a byte (two nibbles) being transmitted.
*   **Channel 3 (Training Sequence Only):** Used during the initial calibration phase. It is *not* used as a separator during data transmission.
*   **Channels 4 through 19 (Data Channels):** These 16 channels are used to transmit 4-bit hexadecimal nibbles (0-F).
    *   Channel 4 represents hex value `0`.
    *   Channel 5 represents hex value `1`.
    *   ...
    *   Channel 19 represents hex value `F`.

### Transmission Protocol:

1.  **Preamble (Channel 1):** A long tone to signal the start of a transmission and allow the receiver to detect the presence of a signal.
2.  **Training Sequence:** A sequence of channels (2, 3, then 4 through 19) is transmitted. The receiver uses these known frequencies to accurately calibrate its detection, mapping each nominal channel ID to the *actual* received frequency. This compensates for speaker/microphone variations and environmental factors.
3.  **Message Header:** After calibration, a structured header is transmitted. This header contains metadata about the data that follows.
    *   **Start Delimiter (1 byte):** `0xFE` - Indicates the beginning of the header.
    *   **Message Type (1 byte):**
        *   `0x00`: Indicates a plain text message (UTF-8 encoded).
        *   `0x01`: Indicates a file transfer.
    *   **If Message Type is `0x00` (Text):**
        *   **Raw Text Data Size (4 bytes, Little-Endian):** The total number of bytes in the raw UTF-8 text message.
    *   **If Message Type is `0x01` (File):**
        *   **Filename Length (2 bytes, Little-Endian):** Length of the UTF-8 encoded filename.
        *   **Filename (Variable bytes):** The UTF-8 encoded original filename.
        *   **Extension Length (2 bytes, Little-Endian):** Length of the UTF-8 encoded file extension.
        *   **File Extension (Variable bytes):** The UTF-8 encoded original file extension.
        *   **Raw File Data Size (4 bytes, Little-Endian):** The total number of bytes in the raw file content.
    *   **End Delimiter (1 byte):** `0xFF` - Indicates the end of the header.
4.  **Payload Data:** Following the header, the raw data (either the UTF-8 text or the binary file content) is transmitted.
5.  **Postamble (Channel 1):** A final tone to mark the end of the entire transmission.

### Byte Transmission Format (for Header and Payload):

For every single byte of the Header and the Payload, it is transmitted using three channel tones:

*   **High Nibble:** The most significant 4 bits of the byte are encoded as a channel ID (4-19).
*   **Low Nibble:** The least significant 4 bits of the byte are encoded as a channel ID (4-19).
*   **Byte Separator (Channel 2):** A dedicated tone to signal the completion of a full byte.

**Example: Transmitting a single byte `0x41` (ASCII 'A')**

Assuming `0x41` is part of the Payload (after the Header):

```
(Header content...)
...
Channel for High Nibble 4 (Ch 8)
Channel for Low Nibble 1 (Ch 5)
Channel 2 (Byte Separator)
...
(Next byte or Postamble)
```

### Efficiency & Robustness:

*   **Fixed Baud Rate:** Each channel tone has a fixed duration (e.g., 30ms), directly controlling the data rate.
*   **4 Bits Per Symbol:** By encoding hexadecimal nibbles, each channel tone carries 4 bits of information (as opposed to 1 bit in simple binary FSK), improving efficiency.
*   **Calibration Tones:** The training sequence allows the receiver to precisely map channel IDs to the actual frequencies detected, significantly improving accuracy and robustness against audio hardware variations.
*   **Byte Separators for Framing:** Channel 2 acts as a crucial frame delimiter, allowing the receiver to properly segment the incoming stream into individual bytes and re-synchronize if a tone is missed.
*   **Automatic Nibble Recovery:** If the receiver detects a Byte Separator (Channel 2) when it was expecting the Low Nibble of a byte, it will automatically assume the missing Low Nibble is a duplicate of the High Nibble. This helps recover from minor signal dropouts and prevent full message desynchronization (warnings for this are suppressed in the console for cleaner logs).
*   **Structured Data Transfer:** The header allows for reliable transmission of different message types (text vs. files) with critical metadata like filename, extension, and exact data size, enabling accurate reconstruction on the receiver side.
*   **No "0" Transmission Overhead:** Unlike bit-position methods, where silent channels might represent "0" bits, this protocol explicitly sends a tone for every nibble, ensuring consistent signal presence.
*   **No Precise Timing Required (within symbol):** The duration is important, but the exact phase or subtle timing variations within a single tone's duration are not critical for decoding, making it more robust than phase-based modulation schemes.

### Current State & Future Ideas:

*   The system currently transmits data in its **raw, uncompressed form**.
*   **No Forward Error Correction (FEC):** While the duplicate nibble recovery helps, robust error correction codes could be added for highly noisy environments.
*   **Compression:** Implementing data compression (e.g., Deflate/zlib) before transmission could significantly increase effective data rates for compressible data.
*   **User Interface:** Both transmitter (web-based) and receiver (Python command-line) are functional.
*   **Community Feedback/Ideas:** Ideas and contributions are always welcome for further improvements in areas like:
    *   Advanced error correction.
    *   More efficient modulation (though more complex for audio).
    *   Cross-platform receiver applications.
    *   Performance optimization.

---
