import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo
    import torch
    from pathlib import Path
    import torch.nn.functional as F
    from einops import rearrange, reduce, repeat

    return F, Path, plt, reduce, torch


@app.cell
def _(Path):
    words = Path("names.txt").read_text().splitlines()
    print(f"{len(words)} words found")

    chars = sorted(set("".join(words)))
    print(f"{len(chars)} unique characters found")

    char_to_i = {ch: i + 1 for i, ch in enumerate(chars)}
    char_to_i["."] = 0

    i_to_char = {i: ch for ch, i in char_to_i.items()}
    return char_to_i, i_to_char, words


@app.cell
def _(char_to_i, torch, words):
    N = torch.zeros(27, 27, dtype=torch.int32)

    for word in words:
        word_chars = [".", *word, "."]
        for ch1, ch2 in zip(word_chars, word_chars[1:]):
            i1 = char_to_i[ch1]
            i2 = char_to_i[ch2]
            N[i1, i2] += 1
    return (N,)


@app.cell
def _(N, i_to_char, plt):
    plt.figure(figsize=(12, 12))
    plt.imshow(N, cmap="Blues")

    for i in range(27):
        for j in range(27):
            bigram_str = i_to_char[i] + i_to_char[j]
            plt.text(j, i, bigram_str, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")

    plt.axis("off")
    plt.show()
    return


@app.cell
def _(N, i_to_char, torch):
    generator = torch.Generator().manual_seed(0)

    P = N.float()
    P = P / P.sum(1, keepdim=True)


    def generate_word():
        word = []
        next_char_index = 0

        while True:
            next_char_index = torch.multinomial(P[next_char_index], num_samples=1, generator=generator).item()

            if next_char_index == 0:
                break

            word.append(i_to_char[next_char_index])

        return "".join(word)


    for _ in range(10):
        print(generate_word())
    return P, generator


@app.cell
def _(P, char_to_i, torch, words):
    def calculate_loss():
        logprobs = []

        for word in words:
            word_chars = [".", *word, "."]

            for ch1, ch2 in zip(word_chars, word_chars[1:]):
                i1 = char_to_i[ch1]
                i2 = char_to_i[ch2]

                p_bigram = P[i1, i2]
                logp_bigram = torch.log(p_bigram)
                logprobs.append(logp_bigram)

        return -sum(logprobs) / len(logprobs)


    calculate_loss()
    return


@app.cell
def _(char_to_i, words):
    def build_training_data():
        x_train = []
        y_train = []

        for word in words:
            chars = [".", *word, "."]
            for prev_char, next_char in zip(chars, chars[1:]):
                x_train.append(char_to_i[prev_char])
                y_train.append(char_to_i[next_char])

        return (x_train, y_train)


    x_train, y_train = build_training_data()
    return x_train, y_train


@app.cell
def _(F, generator, reduce, torch, x_train, y_train):
    generator.manual_seed(0)

    W = torch.randn((27, 27), generator=generator, requires_grad=True)
    x_enc = F.one_hot(torch.tensor(x_train), num_classes=27).float()

    loss = None

    for i in range(100):
        # Forward pass
        logprobs = (x_enc @ W).exp()
        probs = logprobs / reduce(logprobs, "x logprobs -> x 1", "sum")
        loss = -probs[torch.arange(len(x_train)), y_train].log().mean()

        W.grad = None
        loss.backward()

        with torch.no_grad():
            W -= 100 * W.grad

        if (i + 1) % 10 == 0:
            print(f"epoch {i + 1}: {loss.item()}")
    return (W,)


@app.cell
def _(F, W, char_to_i, generator, i_to_char, reduce, torch):
    generator.manual_seed(0)


    def generate_nn_word():
        current_char = "."
        chars = []

        while True:
            current_char_i = char_to_i[current_char]
            current_char_enc = F.one_hot(torch.tensor([current_char_i]), num_classes=27).float()
            logprobs = (current_char_enc @ W).exp()
            probs = (logprobs / reduce(logprobs, "x logprobs -> x 1", "sum"))[0]

            next_char_i = torch.multinomial(probs, num_samples=1, generator=generator).item()
            next_char = i_to_char[next_char_i]

            if next_char == ".":
                break

            chars.append(next_char)
            current_char = next_char

        return "".join(chars)


    [generate_nn_word() for _ in range(5)]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
