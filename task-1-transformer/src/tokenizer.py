"""字符级分词器（中文情感分类无需预训练模型，按字切分即可）。"""


PAD, UNK = "<pad>", "<unk>"


class CharTokenizer:
    def __init__(self, texts=None):
        self.stoi = {PAD: 0, UNK: 1}
        if texts:
            for text in texts:
                for ch in str(text):
                    if ch not in self.stoi:
                        self.stoi[ch] = len(self.stoi)
        self.itos = {i: s for s, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.stoi)

    @property
    def pad_id(self):
        return self.stoi[PAD]

    def encode(self, text):
        unk = self.stoi[UNK]
        return [self.stoi.get(ch, unk) for ch in str(text)]

    def decode(self, ids):
        return "".join(self.itos.get(int(i), UNK) for i in ids)

    @classmethod
    def from_vocab(cls, vocab):
        tok = cls()
        tok.stoi = dict(vocab)
        tok.itos = {i: s for s, i in tok.stoi.items()}
        return tok
