"""Task 4 加分实验：chunk_size 扫描 + reranker 消融。"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GOLD_PATH = ROOT / "data" / "gold_qa.jsonl"


def load_gold():
    return [json.loads(l) for l in GOLD_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def normalize(text):
    return re.sub(r"\s+", "", str(text))


def evaluate(retriever, use_rerank=False, reranker=None, candidate_k=20, output_k=10):
    """在 gold QA 上计算 Recall@k 和 MRR。"""
    gold = load_gold()
    hits = {1: 0, 3: 0, 5: 0, 10: 0}
    rr = []
    for item in gold:
        anchors = [(a, normalize(a)) for a in item.get("gold_anchors", [])]
        results = retriever.retrieve(item["question"], k=candidate_k)
        if use_rerank and reranker:
            results = reranker.rerank(item["question"], results, top_k=output_k)
        else:
            results = results[:output_k]
        matched_rank = None
        for rank, r in enumerate(results, 1):
            text = normalize(r.get("text", ""))
            for raw, a in anchors:
                if a and a in text:
                    matched_rank = rank
                    break
            if matched_rank:
                break
        if matched_rank:
            for k in hits:
                if matched_rank <= k:
                    hits[k] += 1
            rr.append(1 / matched_rank)
        else:
            rr.append(0)
    n = len(gold)
    return {
        "recall_at_1": round(hits[1] / n, 3),
        "recall_at_3": round(hits[3] / n, 3),
        "recall_at_5": round(hits[5] / n, 3),
        "recall_at_10": round(hits[10] / n, 3),
        "mrr": round(sum(rr) / n, 3),
        "n": n,
    }


def build_and_eval_chunk_size(chunk_size, overlap=48):
    """用指定 chunk_size 重建索引并评测。"""
    from src.indexer import build_index
    from src.retriever import Retriever
    build_index(chunk_size=chunk_size, overlap=overlap, force=True)
    retriever = Retriever()
    return evaluate(retriever, use_rerank=False)


def eval_rerank():
    """用默认索引评测有/无 reranker。"""
    from src.retriever import Retriever
    from src.reranker import Reranker
    retriever = Retriever()
    no_rerank = evaluate(retriever, use_rerank=False)
    reranker = Reranker()
    with_rerank = evaluate(retriever, use_rerank=True, reranker=reranker)
    return no_rerank, with_rerank


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chunk", "rerank", "all"], default="all")
    args = parser.parse_args()

    results = {}

    if args.mode in ("chunk", "all"):
        print("=" * 60)
        print("S1: chunk_size 扫描")
        print("=" * 60)
        for cs in [128, 256, 512, 1024]:
            print(f"\n--- chunk_size={cs}, overlap=48 ---")
            m = build_and_eval_chunk_size(cs)
            results[f"chunk_{cs}"] = m
            print(json.dumps(m, indent=2))
        # 恢复默认索引
        print("\n恢复默认 chunk_size=256...")
        build_and_eval_chunk_size(256)

    if args.mode in ("rerank", "all"):
        print("\n" + "=" * 60)
        print("S2: Reranker 消融")
        print("=" * 60)
        no_r, with_r = eval_rerank()
        results["no_reranker"] = no_r
        results["with_reranker"] = with_r
        print(f"\n无 reranker: {json.dumps(no_r)}")
        print(f"有 reranker: {json.dumps(with_r)}")

    out = ROOT / "eval" / "ablation_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存到 {out}")
