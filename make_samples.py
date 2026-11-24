# make_samples.py - makes sample files for testing and the data used to train the model
import os
import random

SYLLABLES = ["ba", "ke", "ri", "to", "su", "ma", "ne", "lo", "pi", "da", "vo", "re", "chi", "an", "el", "or"]
COMMON = ["the", "of", "and", "to", "in", "is", "that", "for", "with", "on"]


def make_vocab(topic):
    r = random.Random(topic)
    return ["".join(r.choice(SYLLABLES) for _ in range(r.randint(1, 3))) for _ in range(150)]


def text_chunk(size):
    vocab = make_vocab(random.randint(0, 20)) + COMMON * 5
    out = ""
    while len(out) < size:
        words = [random.choice(vocab) for _ in range(random.randint(6, 16))]
        out += " ".join(words).capitalize() + random.choice([". ", ". ", "? ", ".\n"])
    return out[:size].encode()


def csv_chunk(size):
    cities = ["Delhi", "Mumbai", "Pune", "Chennai", "Jaipur", "Surat"]
    out = "id,name,city,amount\n"
    i = random.randint(1, 100000)
    while len(out) < size:
        out += "%d,%s,%s,%.2f\n" % (i, random.choice(SYLLABLES).capitalize() + random.choice(SYLLABLES),
                                    random.choice(cities), random.random() * 5000)
        i += 1
    return out[:size].encode()


def log_chunk(size):
    out = ""
    while len(out) < size:
        out += "2024-%02d-%02d %02d:%02d:%02d %s request id=%d took %dms\n" % (
            random.randint(1, 12), random.randint(1, 28), random.randint(0, 23), random.randint(0, 59),
            random.randint(0, 59), random.choice(["INFO", "INFO", "WARN", "ERROR"]),
            random.randint(1, 99999), random.randint(1, 800))
    return out[:size].encode()


def binary_chunk(size):
    return bytes(random.getrandbits(8) for _ in range(size))


def random_chunk(size):
    return random.choice([text_chunk, text_chunk, csv_chunk, log_chunk, binary_chunk])(size)


def modify(data, percent):
    """Make a slightly changed copy: replace / insert / delete a few small pieces."""
    data = bytearray(data)
    edits = max(1, int(len(data) * percent / 100 / 10))
    for _ in range(edits):
        if len(data) < 50:
            break
        pos = random.randint(0, len(data) - 20)
        length = random.randint(3, 15)
        new = bytes(random.choice(data) for _ in range(length))
        kind = random.randint(0, 2)
        if kind == 0:
            data[pos:pos + length] = new       # replace
        elif kind == 1:
            data[pos:pos] = new                # insert
        else:
            del data[pos:pos + length]         # delete
    return bytes(data)


def write(folder, name, data):
    with open(os.path.join(folder, name), "wb") as f:
        f.write(data)


def make_sample_files(folder="sample_data"):
    random.seed(7)
    os.makedirs(folder, exist_ok=True)
    report = b"".join(text_chunk(1000) for _ in range(200))
    write(folder, "report.txt", report)
    write(folder, "report_copy.txt", report)                         # exact duplicate, other name
    write(folder, "report_edited.txt", modify(report, 1))            # slightly modified
    write(folder, "report_shifted.txt", b"NEW FIRST PARAGRAPH\n\n" + report)  # text added at the top
    write(folder, "other_notes.txt", b"".join(text_chunk(1000) for _ in range(100)))  # different file
    sales = csv_chunk(120000)
    write(folder, "sales.csv", sales)
    write(folder, "sales_edited.csv", modify(sales, 2))
    write(folder, "server.log", log_chunk(80000))
    write(folder, "random.bin", binary_chunk(60000))
    block = binary_chunk(4096)
    write(folder, "repeated.bin", block * 30 + binary_chunk(3000))   # repeated chunks
    write(folder, "empty.txt", b"")
    print("sample files created in", folder)


if __name__ == "__main__":
    make_sample_files()
