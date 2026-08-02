import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import torch
    import torch.nn as nn
    from torch.nn import functional as F

    return F, Path, nn, torch


@app.cell
def _(Path):
    text = Path("./shakespeare.txt").read_text(encoding="utf-8")
    print(f"Text has {len(text):,} chars")
    return (text,)


@app.cell
def _(text):
    chars = sorted(set(text))
    print(f"VOCAB: {''.join(chars)}")
    print(f"{len(chars)} chars")
    return (chars,)


@app.cell
def _(chars):
    char_to_token = {c: t for t, c in enumerate(chars)}
    token_to_char = {t: c for t, c in enumerate(chars)}
    encode = lambda s: [char_to_token[c] for c in s]
    decode = lambda l: "".join([token_to_char[t] for t in l])
    vocab_size = len(token_to_char)

    print(encode("hello"))
    print(decode(encode("hello")))
    return decode, encode, vocab_size


@app.cell
def _(decode, encode, text, torch):
    data = torch.tensor(encode(text), dtype=torch.long)
    print(data.shape, data.dtype)
    (data[:10], decode(data[:10].tolist()))
    return (data,)


@app.cell
def _(data):
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train data: {len(train_data):,} tokens")
    print(f"Validation data: {len(val_data):,} tokens")
    return train_data, val_data


@app.cell
def _(torch, train_data, val_data):
    torch.manual_seed(1337)

    batch_size = 4
    context_length = 8

    def get_batch(split="train"):
        data = train_data if split == "train" else val_data
        start_indices = torch.randint(len(data) - context_length, (batch_size,))
        # print(start_indices)
        x = torch.stack([data[i : i + context_length] for i in start_indices])
        y = torch.stack([data[i + 1 : i + context_length + 1] for i in start_indices])
        return x, y

    x_batch, y_batch = get_batch("train")
    print(x_batch)
    print(y_batch)
    return x_batch, y_batch


@app.cell
def _(F, nn, vocab_size, x_batch, y_batch):
    class BigramLanguageModel(nn.Module):
        def __init__(self, vocab_size):
            # TODO: Why vocab_size * vocab_size?
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        def forward(self, x, y):
            logits = self.token_embedding_table(x)
            B, T, C = logits.shape
            print(x.shape, logits.shape, y.shape)
            loss = F.cross_entropy(logits.view(B*T, C), y.view(-1))
            return logits, loss

        def generate(self, x, max_tokens=100):
            # TODO: Implement this
            return None

    m = BigramLanguageModel(vocab_size)
    logits, loss = m(x_batch, y_batch)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
