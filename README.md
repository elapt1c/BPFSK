# BPFSK
bit position frequency shift keying

# HOW IT WORKS:
The string is converted to ASCII binary.
each byte is cycled through, and every bit that is on will be played from most significant bit to least significant bit.
|1|2|3|4|5|6|7|8|9|10|
    |0|1|0|1|0|1|0|1 |
in that example, 01010101 would be (4,6,8,10)
channel 1 is reserved for preamble/postamble and channel 2 is reserved for the byte seperator, which tells when a byte is done being played
for the example above, the single byte 01010101 would be played as

1 (preamble)
4
6
8
10
2 (byte seperator)
1 (postamble)

Each channel is a frequency evenly spaced from the rest. for example, channel 1 could be 1000 khz, channel 2 could be 1100 khz, channel 3 could be 1200 khz, etc.

This is efficient since you don't have to transmit the 0's and precise timing isnt required. The downside is that precice frequency control is required, though calibration tones could be added to map each channel to the correct frequency on the receiver side.

Community feedback/ideas are greatly appreciated, and any help with code and other community provided additions are needed.
We need a receiver and a way to sync it up with each transmitted channel.
