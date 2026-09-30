"""BGE 向量检索：加载 FAISS 索引 + 查询编码 + top-k 召回。"""
import pickle
from pathlib import Path
from typing import List, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
INDEX_DIR = DATA_DIR / "index"
EMBED_MODEL = MODELS_DIR / "bge-small-zh-v1.5"
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class Retriever:
    def __init__(self, index_dir: Path = INDEX_DIR, embed_model_path: Path = EMBED_MODEL):
        import faiss
        from sentence_transformers import SentenceTransformer
        
        index_file = index_dir / "faiss.index"
        chunks_file = index_dir / "chunks.pkl"
        
        if not index_file.exists() or not chunks_file.exists():
            from src.indexer import build_index
            build_index()
        
        self.index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            self.chunks = pickle.load(f)
        self.model = SentenceTransformer(str(embed_model_path), device="cpu")
    
    def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        """返回 top-k 相关 chunk，每个 dict 含 text / score / source。"""
        query_text = BGE_QUERY_PREFIX + query
        emb = self.model.encode([query_text], normalize_embeddings=True)
        emb = emb.astype("float32")
        
        scores, indices = self.index.search(emb, min(k, len(self.chunks)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append({
                "text": self.chunks[idx],
                "score": float(score),
                "source": f"kb.pdf#chunk_{idx}",
            })
        return results
