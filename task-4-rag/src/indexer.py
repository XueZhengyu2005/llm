"""从 PDF 抽取文本 → chunk → BGE embedding → FAISS 索引。"""
import json
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
INDEX_DIR = DATA_DIR / "index"
PDF_PATH = DATA_DIR / "kb.pdf"
EMBED_MODEL = MODELS_DIR / "bge-small-zh-v1.5"

CHUNK_SIZE = 256
CHUNK_OVERLAP = 48


def extract_pdf_text(pdf_path: Path = PDF_PATH) -> str:
    """从 PDF 提取全部文本。"""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(f"[PAGE {i+1}]\n{text}")
    return "\n\n".join(pages)


def build_index(chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
                force: bool = False) -> None:
    """构建 FAISS 索引并保存。"""
    import faiss
    from sentence_transformers import SentenceTransformer
    from src.chunker import chunk_text

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_file = INDEX_DIR / "faiss.index"
    chunks_file = INDEX_DIR / "chunks.pkl"
    
    if index_file.exists() and chunks_file.exists() and not force:
        print("索引已存在，跳过构建。如需重建请加 --force")
        return
    
    print("1. 提取 PDF 文本...")
    text = extract_pdf_text()
    print(f"   共 {len(text)} 字符")
    
    print("2. Chunking...")
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    print(f"   共 {len(chunks)} 个 chunk")
    
    print("3. 加载 BGE embedding 模型...")
    model = SentenceTransformer(str(EMBED_MODEL), device="cpu")
    
    print("4. 编码 chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True,
                              normalize_embeddings=True, batch_size=64)
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"   embedding shape: {embeddings.shape}")
    
    print("5. 构建 FAISS 索引...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 内积 = cosine（已归一化）
    index.add(embeddings)
    
    faiss.write_index(index, str(index_file))
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)
    
    print(f"✅ 索引已保存到 {INDEX_DIR}/ ({len(chunks)} chunks, dim={dim})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="强制重建索引")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()
    build_index(chunk_size=args.chunk_size, overlap=args.overlap, force=args.force)
