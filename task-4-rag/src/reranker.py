"""BGE reranker 精排。"""
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RERANK_MODEL = MODELS_DIR / "bge-reranker-base"


class Reranker:
    def __init__(self, model_path: Path = RERANK_MODEL):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(str(model_path), max_length=512, device="cpu")
    
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """对检索结果重排序，返回 top_k。"""
        if not results:
            return []
        pairs = [[query, r["text"]] for r in results]
        scores = self.model.predict(pairs, show_progress_bar=False)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]
