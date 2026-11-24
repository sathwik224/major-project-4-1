# chunking.py - split a file into chunks (fixed size or content defined)
import random

# random numbers for the gear hash (same every run, so chunks are the same every run)
random.seed(42)
GEAR = [random.getrandbits(32) for _ in range(256)]


def fixed_chunks(data, size=4096):
    return [data[i:i + size] for i in range(0, len(data), size)]


def cdc_chunks(data, avg_size=4096):
    """Content defined chunking with a gear hash (the idea behind FastCDC).

    A chunk ends when the hash of the last ~32 bytes matches a pattern, so the
    cut points move together with the content. Because of this, inserting some
    bytes at the start of a file only changes the first chunk or two.
    """
    min_size = avg_size // 4
    max_size = avg_size * 4
    bits = avg_size.bit_length() - 1          # 4096 -> 12 bits
    mask = ((1 << bits) - 1) << (32 - bits)   # look at the top bits of the hash

    chunks = []
    start = 0
    n = len(data)
    while start < n:
        end = min(start + max_size, n)
        pos = start + max(min_size - 32, 0)   # skip the first part (min size), only warm up the hash
        h = 0
        cut = end
        while pos < end:
            h = ((h << 1) + GEAR[data[pos]]) & 0xFFFFFFFF
            pos += 1
            if pos - start >= min_size and (h & mask) == 0:
                cut = pos
                break
        chunks.append(data[start:cut])
        start = cut
    return chunks


def split_file(data, mode="cdc", size=4096):
    if mode == "fixed":
        return fixed_chunks(data, size)
    return cdc_chunks(data, size)
