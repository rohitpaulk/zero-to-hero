from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

# --- Data loading ---

text = Path("./shakespeare.txt").read_text(encoding="utf-8")
print(f"Text has {len(text):,} chars")

chars = sorted(set(text))
print(f"VOCAB: {''.join(chars)}")
print(f"{len(chars)} chars")

char_to_token = {c: t for t, c in enumerate(chars)}
token_to_char = {t: c for t, c in enumerate(chars)}
encode = lambda s: [char_to_token[c] for c in s]
decode = lambda l: "".join([token_to_char[t] for t in l])
vocab_size = len(token_to_char)

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"Train data: {len(train_data):,} tokens")
print(f"Validation data: {len(val_data):,} tokens")

# --- Batching ---

torch.manual_seed(1337)

batch_size = 32
context_length = 8


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    start_indices = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in start_indices])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in start_indices])
    return x, y


# --- Model ---


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
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# --- Training ---

for step_index in range(12000):
    x_batch, y_batch = get_batch("train")
    logits, loss = m(x_batch, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step_index % 1000 == 0:
        print(f"step {step_index}: loss {loss.item():.4f}")

print(f"final loss: {loss.item():.4f}")

# --- Generation ---

out = m.generate(torch.zeros(1, 1, dtype=torch.long))
print(decode(out[0].tolist()))
