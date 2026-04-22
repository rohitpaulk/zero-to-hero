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

    return Path, plt, torch


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


if __name__ == "__main__":
    app.run()
