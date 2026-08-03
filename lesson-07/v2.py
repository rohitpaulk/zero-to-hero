from pathlib import Path

import plotext as plt
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
embedding_size = 32


def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    start_indices = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in start_indices])
    y = torch.stack([data[i + 1 : i + context_length + 1] for i in start_indices])
    return x, y


# --- Model ---


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()

        self.head_size = head_size

        self.key_weights = nn.Linear(embedding_size, head_size, bias=False)
        self.query_weights = nn.Linear(embedding_size, head_size, bias=False)
        self.value_weights = nn.Linear(embedding_size, head_size, bias=False)

        self.register_buffer("attention_mask", torch.tril(torch.ones(context_length, context_length)) == 0)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key_weights(x)  # B, T, head_size
        q = self.query_weights(x)  # B, T, head_size

        attn_weights = q @ k.transpose(-1, -2) / self.head_size**0.5  # B, T, T
        attn_weights.masked_fill_(self.attention_mask[:T, :T], float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)

        v = self.value_weights(x)  # B, T, head_size
        return attn_weights @ v  # B, T, head_size


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(context_length, embedding_size)
        self.attention_head = SelfAttentionHead(32)
        self.lm_head = nn.Linear(embedding_size, vocab_size)

    def forward(self, x, y=None):
        B, T = x.shape
        token_embeddings = self.token_embedding_table(x)  # (B, T, n_embd)
        position_embeddings = self.position_embedding_table(torch.arange(T, device=x.device))  # (T, n_embd)
        x = token_embeddings + position_embeddings  # (B, T, n_embd)
        x = self.attention_head(x)  # (B, T, head_size)
        logits = self.lm_head(x)  # (B, T, vocab_size)

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


m = BigramLanguageModel()
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# --- Training ---

total_steps = 12000
plot_every = 200  # redraw the live chart every N steps


def draw_loss_chart(losses, step_index):
    # Average every 10 steps to smooth the curve
    smoothed = torch.tensor(losses).reshape(-1, 10).mean(dim=1)

    plt.clt()  # clear the terminal
    plt.cld()  # clear previous plot data
    plt.plot(smoothed.tolist())
    plt.limit_size(False, False)  # don't clamp to detected terminal size
    plt.plotsize(100, 25)
    plt.title(f"Training loss — step {step_index + 1:,}/{total_steps:,} (loss {losses[-1]:.4f})")
    plt.xlabel("step (x10)")
    plt.ylabel("loss")
    plt.show()


losses = []

for step_index in range(total_steps):
    x_batch, y_batch = get_batch("train")
    logits, loss = m(x_batch, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (step_index + 1) % plot_every == 0:
        draw_loss_chart(losses, step_index)

print(f"final loss: {losses[-1]:.4f}")

# --- Generation ---

out = m.generate(torch.zeros(1, 1, dtype=torch.long))
print(decode(out[0].tolist()))
