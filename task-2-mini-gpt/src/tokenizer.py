"""手写字节级 BPE tokenizer：中文按 UTF-8 字节回退，保证 encode/decode 无损。"""
import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    def __init__(self):
        # id 0-255 对应单字节；merge 出的 token 依次编号
        self.vocab = [bytes([i]) for i in range(256)]
        self.merge_rules = {}  # (left_id, right_id) -> new_id

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _add_merge(self, left, right):
        new_id = len(self.vocab)
        self.vocab.append(self.vocab[left] + self.vocab[right])
        self.merge_rules[(left, right)] = new_id
        return new_id

    def train(self, text, num_merges=512):
        """在给定语料上迭代合并出现次数最高的相邻 token 对。"""
        seq = list(text.encode("utf-8"))
        for _ in range(num_merges):
            counts = Counter(zip(seq, seq[1:]))
            if not counts:
                break
            # 频次相同按 token id 排序，保证结果可复现
            best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            if counts[best] < 2:
                break
            self._add_merge(*best)
            seq = self._apply_merge(seq, best)
        return self

    def _apply_merge(self, seq, pair):
        new_id = self.merge_rules[pair]
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def encode(self, text):
        seq = list(text.encode("utf-8"))
        while len(seq) >= 2:
            # 每轮优先应用「最早学到的 merge」（BPE 标准语义）
            best, best_rank = None, None
            for pair in zip(seq, seq[1:]):
                rank = self.merge_rules.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best, best_rank = pair, rank
            if best is None:
                break
            seq = self._apply_merge(seq, best)
        return seq

    def decode(self, ids):
        data = b"".join(self.vocab[int(i)] for i in ids)
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        payload = {"merges": [list(p) for p in self.merge_rules]}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path):
        tok = cls()
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for left, right in payload["merges"]:
            tok._add_merge(int(left), int(right))
        return tok
