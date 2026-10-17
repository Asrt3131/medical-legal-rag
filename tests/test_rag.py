import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from rag import build_index, search

def test_build_and_search():
    # با یک PDF تستی
    index, chunks = build_index("data/sample.pdf")
    assert index.ntotal > 0
    assert len(chunks) > 0

    results = search(index, chunks, "تست سوال", k=2)
    assert len(results) == 2
    assert "text" in results[0]

if __name__ == "__main__":
    test_build_and_search()
    print("✅ تست‌ها پاس شدند")