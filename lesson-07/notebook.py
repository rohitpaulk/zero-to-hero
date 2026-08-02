import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
    import matplotlib.pyplot as plt

    return F, Path, nn, plt, torch


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

    batch_size = 32
    context_length = 8

    def get_batch(split="train"):
        data = train_data if split == "train" else val_data
        start_indices = torch.randint(len(data) - context_length, (batch_size,))
        # print(start_indices)
        x = torch.stack([data[i : i + context_length] for i in start_indices])
        y = torch.stack([data[i + 1 : i + context_length + 1] for i in start_indices])
        return x, y

    return context_length, get_batch


@app.cell
def _(F, context_length, nn, torch, vocab_size):
    class BigramLanguageModel(nn.Module):
        def __init__(self, vocab_size):
            # TODO: Why vocab_size * vocab_size?
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        def forward(self, x, y=None):
            logits = self.token_embedding_table(x)

            if y is not None:
                B, T, C = logits.shape
                loss = F.cross_entropy(logits.view(B * T, C), y.view(-1))
            else:
                loss = None

            return logits, loss

        def generate(self, x, max_tokens=100):
            result = x

            for _ in range(max_tokens):
                logits, _ = self.forward(x)
                logits = logits[:, -1, :]  # Only look at the last time step
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, 1)

                result = torch.cat((result, next_tokens), dim=1)
                x = result[:, -context_length:]

            return result

    m = BigramLanguageModel(vocab_size)
    return (m,)


@app.cell
def _(m, torch):
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    return (optimizer,)


@app.cell
def _(get_batch, m, optimizer, plt, torch):
    losses = []

    for step_index in range(12000):        
        x_batch, y_batch = get_batch("train")
        logits, loss = m(x_batch, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    losses = torch.tensor(losses)
    plt.plot(losses.reshape(-1, 10).mean(axis=1))
    return


@app.cell
def _(decode, m, torch):
    out = m.generate(torch.zeros(1, 1, dtype=torch.long))
    print(decode(out[0].tolist()))
    return


if __name__ == "__main__":
    app.run()
