from rag_chunk_optimize.chunker import chunk_text
from rag_chunk_optimize.optimizer import optimize, _eval
from rag_chunk_optimize.types import ChunkConfig

class TestChunker:
    def test_sentence(self):
        c = chunk_text("A. B. C.", ChunkConfig("sentence", 500, 0))
        assert len(c) >= 1

    def test_paragraph(self):
        c = chunk_text("Para one.\n\nPara two.\n\nPara three.", ChunkConfig("paragraph", 20, 0))
        assert len(c) >= 2

    def test_fixed(self):
        c = chunk_text("abcdefghij", ChunkConfig("fixed", 4, 0))
        assert len(c) == 3

    def test_overlap(self):
        c = chunk_text("First. Second. Third.", ChunkConfig("sentence", 60, 20))
        assert len(c) >= 1


class TestOptimizer:
    def test_eval(self):
        r = _eval(ChunkConfig("sentence", 100, 0), {"d1": "Python AI. ML is popular."}, ["Python", "ML"])
        assert r.score > 0

    def test_optimize(self):
        r = optimize({"d1": "Python. ML. Data."}, ["Python", "data"], ["sentence"], [100, 300], [0])
        assert r.total_tested >= 2

    def test_best_highest(self):
        r = optimize({"d1": "A. B. C. D. E. F."}, ["A B", "D E"], ["sentence"], [50, 200], [0])
        assert r.best is not None
        assert r.best.score >= r.trials[-1].score

    def test_summary(self):
        r = optimize({"d1": "X. Y."}, ["X"], ["sentence"], [100], [0])
        assert "Tested" in r.summary()
