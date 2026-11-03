"""memory_search queries written as ordinary questions reach the archive.

A trailing ``?`` turned ``Quillmark?`` into ``Quillmark?*``, which FTS5
rejects; store.py logged the error and returned [], so the search looked
like "nothing found". Runs against a real SQLite archive, no LLM.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class FtsQueryTests(unittest.TestCase):
    def setUp(self) -> None:
        from plugins.autobio.store import get_store

        self.tmp = tempfile.mkdtemp()
        self.store = get_store(self.tmp, "fts-query")
        docs = {
            "known": ("Rurik Pellegrin uses Quillmark. Память хранит историю. "
                      "Version v1.2 supports C++."),
            "pastry": "An éclair for dessert.",
            "noise1": "Somewhere elsewhere.",
            "noise2": "A clairvoyant cafeteria.",
            "private": "Glyph  marks the spot.",
        }
        for pos, (sid, text) in enumerate(docs.items()):
            self.store.upsert_message(sid, {"role": "user", "content": text},
                                      position=pos)

    def tearDown(self) -> None:
        from plugins.autobio.store import close_store

        close_store(self.tmp, "fts-query")
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _sources(self, query: str) -> list:
        from plugins.autobio.search import search_all

        return [h["source"] for h in search_all(self.store, query)]

    def test_question_shaped_queries_find_the_message(self) -> None:
        for query in [
            "Quillmark?", "Quillmark!", "Quillmark.", "Quillmark:",
            "Quillmark /", "Quillmark [test]", "Память?", "v1.2?", "C++?",
            '"Quillmark"', "Quillmark OR missing",
        ]:
            with self.subTest(query=query):
                self.assertIn("known", self._sources(query))

    def test_queries_without_words_return_nothing(self) -> None:
        for query in ["", "? ! / :", "totallyabsentword?"]:
            with self.subTest(query=query):
                self.assertEqual([], self._sources(query))

    def test_decomposed_accent_stays_one_word(self) -> None:
        sources = self._sources("éclair?")
        self.assertIn("pastry", sources)
        self.assertNotIn("noise1", sources)
        self.assertNotIn("noise2", sources)

    def test_private_use_character_is_searchable(self) -> None:
        self.assertIn("private", self._sources(""))

    def test_query_is_not_renormalized(self) -> None:
        # The tokenizer sees the query text as typed, like the indexed text.
        self.store.upsert_message("hangul", {"role": "user", "content": "가"},
                                  position=10)
        self.store.upsert_message("dotted", {"role": "user", "content": "ộ"},
                                  position=11)
        self.assertNotIn("hangul", self._sources("가"))
        self.assertNotIn("dotted", self._sources("ộ"))

    def test_word_split_matches_unicode61(self) -> None:
        import sqlite3

        from plugins.autobio.search import _sanitize_fts_query

        db = sqlite3.connect(":memory:")
        db.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize="
                   "'unicode61 remove_diacritics 1')")
        db.execute("CREATE VIRTUAL TABLE v USING fts5vocab(t, instance)")

        def tokens(text: str) -> list:
            db.execute("DELETE FROM t")
            db.execute("INSERT INTO t(rowid, x) VALUES (1, ?)", (text,))
            rows = db.execute("SELECT term FROM v ORDER BY offset").fetchall()
            return [r[0] for r in rows]

        samples = [
            "Quillmark? C++ v1.2 [test] a/b", "éclair café",
            "fooःbar foo⃝bar", " 가 ộ",
            "Память хранит историю.", "snake_case x² ١٢",
            "日本語テキスト \U0001f600 emoji",
        ]
        for text in samples:
            with self.subTest(text=text):
                q = _sanitize_fts_query(text)
                ours = [part[1:-2] for part in q.split(" OR ")] if q else []
                self.assertEqual(tokens(text), [t for w in ours for t in tokens(w)])
                for w in ours:
                    self.assertEqual(1, len(tokens(w)), w)
        db.close()

    def test_spacing_mark_separates_words(self) -> None:
        # unicode61 splits on Mc/Me marks, so each side is its own OR term.
        self.store.upsert_message("foo", {"role": "user", "content": "foo"},
                                  position=12)
        self.store.upsert_message("bar", {"role": "user", "content": "bar"},
                                  position=13)
        for query in ["fooःbar", "foo⃝bar"]:
            with self.subTest(query=query):
                sources = self._sources(query)
                self.assertIn("foo", sources)
                self.assertIn("bar", sources)


if __name__ == "__main__":
    unittest.main()
