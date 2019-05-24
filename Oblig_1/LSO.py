# -*- coding: utf-8 -*-

# List of all acceptable symbols.
symbols = list("ABCDEFGHIJKLMNOPQRSTUVWXYZÆØÅ0123456789 -,.!?".lower())


def LSO(msg, shift=1):
    """Left Shift Operation."""

    # Make the string a list of lower case chars.
    msg = list(msg.lower())

    # For each letter in message.
    for i in range(len(msg)):

        # Get index of msg-letter in symbols.
        index = symbols.index(msg[i])
        
        # If not found, return.
        if index == -1:
            return
        
        # LSO swap. Modulate to stay within bounds.
        msg[i] = symbols[(index + shift) % len(symbols)]
    return "".join(msg)


def RSO(msg, shift=1):
    """Right Shift Operation (opposite of LSO)."""
    return LSO(msg, len(symbols) - shift)
