# test_dedup.py - tests for the cases in the project brief      ->   python test_dedup.py
import os
import random
import shutil
import tempfile
import unittest

import database
import storage_manager
import deduplication as dd
import make_samples as ms
import chunking


class DedupTests(unittest.TestCase):
    def setUp(self):
        # every test uses its own empty database and chunk folder
        self.tmp = tempfile.mkdtemp()
        database.DB_PATH = os.path.join(self.tmp, "test.db")
        storage_manager.STORAGE_DIR = os.path.join(self.tmp, "chunks")
        database.init_db()
        self.db = database.get_db()
        random.seed(5)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def upload(self, name, data, **kw):
        kw.setdefault("mode", "fixed")
        return dd.process_file(self.db, name, data, "txt", **kw)

    def text(self, size):
        return b"".join(ms.text_chunk(1000) for _ in range(size // 1000))

    def test_1_identical_files_different_names(self):
        data = self.text(100000)
        a = self.upload("a.txt", data)
        b = self.upload("b_copy.txt", data)
        print("\n1. identical files: second file stored", b["stored"], "bytes, duplicate_of =", b["duplicate_of"])
        self.assertEqual(b["stored"], 0)
        self.assertEqual(b["duplicate_of"], a["file_id"])
        self.assertEqual(dd.rebuild_file(self.db, b["file_id"]), data)

    def test_2_slightly_modified_files(self):
        data = self.text(100000)
        edited = ms.modify(data, 1)
        self.upload("v1.txt", data, use_dl=False)
        hash_only = self.upload("v2.txt", edited, use_dl=False)
        self.db.close()
        self.setUp()   # start again with an empty database for the deep learning run
        self.upload("v1.txt", data)
        with_dl = self.upload("v2.txt", edited)
        print("\n2. modified file: hash only stored", hash_only["stored"], "bytes, with deep learning", with_dl["stored"],
              "bytes (%d chunks merged)" % with_dl["similar"])
        self.assertLess(with_dl["stored"], hash_only["stored"] / 2)
        self.assertEqual(dd.rebuild_file(self.db, with_dl["file_id"]), edited)

    def test_3_completely_different_files(self):
        a = bytes(random.getrandbits(8) for _ in range(40000))
        b = bytes(random.getrandbits(8) for _ in range(40000))
        self.upload("a.bin", a)
        r = self.upload("b.bin", b)
        print("\n3. different files: exact =", r["exact"], "similar =", r["similar"], "stored", r["stored"], "of", r["size"])
        self.assertEqual(r["exact"], 0)
        self.assertEqual(r["similar"], 0)
        self.assertEqual(r["stored"], r["size"])

    def test_4_repeated_chunks(self):
        block = bytes(random.getrandbits(8) for _ in range(4096))
        data = block * 30
        r = self.upload("rep.bin", data, use_dl=False)
        print("\n4. repeated chunks: 30 chunks, exact duplicates =", r["exact"], ", stored", r["stored"], "of", r["size"])
        self.assertEqual(r["exact"], 29)
        self.assertEqual(r["stored"], 4096)
        self.assertEqual(dd.rebuild_file(self.db, r["file_id"]), data)

    def test_5_empty_file(self):
        r = self.upload("empty.txt", b"")
        print("\n5. empty file: chunks =", r["chunks"], ", saved_percent =", r["saved_percent"])
        self.assertEqual(r["chunks"], 0)
        self.assertEqual(dd.rebuild_file(self.db, r["file_id"]), b"")

    def test_6_large_file(self):
        big = self.text(4000000)
        a = self.upload("big.txt", big, mode="cdc", use_dl=False)
        modified = big[:2000000] + b"<<inserted text>>" + big[2000000:]
        b = self.upload("big2.txt", modified, mode="cdc", use_dl=False)
        print("\n6. large file (4 MB): %.1f s, %.2f MB/s; after inserting text in the middle %d of %d chunks were exact duplicates"
              % (a["seconds"], a["throughput"], b["exact"], b["chunks"]))
        self.assertGreater(b["exact"], b["chunks"] * 0.95)
        self.assertEqual(dd.rebuild_file(self.db, b["file_id"]), modified)

    def test_7_reconstruction_and_integrity(self):
        base = self.text(60000)
        files = {"a": base, "b": b"header\n" + base, "c": ms.modify(base, 2), "d": b""}
        ids = {k: self.upload(k + ".txt", v)["file_id"] for k, v in files.items()}
        for k, i in ids.items():
            self.assertEqual(dd.rebuild_file(self.db, i), files[k])
        # damage one stored chunk -> rebuilding must fail, never give wrong data
        row = self.db.execute("SELECT path FROM chunks WHERE kind='full' LIMIT 1").fetchone()
        with open(os.path.join(storage_manager.STORAGE_DIR, row["path"]), "r+b") as f:
            f.write(b"X")
        failed = 0
        for i in ids.values():
            try:
                dd.rebuild_file(self.db, i)
            except dd.IntegrityError:
                failed += 1
        print("\n7. integrity: all files rebuilt identical; after damaging one chunk,", failed, "files report an error")
        self.assertGreater(failed, 0)

    def test_8_delete_is_safe(self):
        base = self.text(50000)
        edited = ms.modify(base, 1)
        a = self.upload("a.txt", base)["file_id"]
        b = self.upload("b.txt", edited)["file_id"]
        dd.delete_file(self.db, a)                       # b may need chunks of a (delta base)
        self.assertEqual(dd.rebuild_file(self.db, b), edited)
        dd.delete_file(self.db, b)
        left = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        print("\n8. delete: chunks left after deleting everything =", left)
        self.assertEqual(left, 0)

    def test_9_cdc_survives_insertion(self):
        data = bytes(random.getrandbits(8) for _ in range(200000))
        shifted = b"xxxxxxxxxxxxxxxxx" + data
        h = lambda ch: {__import__("hashlib").sha256(c).hexdigest() for c in ch}
        cdc = len(h(chunking.cdc_chunks(data)) & h(chunking.cdc_chunks(shifted))) / len(chunking.cdc_chunks(data))
        fixed = len(h(chunking.fixed_chunks(data)) & h(chunking.fixed_chunks(shifted))) / len(chunking.fixed_chunks(data))
        print("\n9. after inserting 17 bytes: CDC keeps %.0f%% of the chunks, fixed size keeps %.0f%%" % (cdc * 100, fixed * 100))
        self.assertGreater(cdc, 0.8)
        self.assertLess(fixed, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=1)
