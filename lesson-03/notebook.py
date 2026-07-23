import marimo

__generated_with = "0.23.14"
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

    return Path, torch


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
def _(char_to_i, i_to_char, torch, words):
    block_size = 4
    X, Y = [], []

    for word in words[:5]:
        context = [char_to_i["."]] * block_size  # Start with "...."

        for char in word + ".":
            char_i = char_to_i[char]
            X.append(context)
            Y.append(char_i)

            # Shift "...." -> "...e", "...e" to "..em"
            context = context[1:] + [char_i]

    for i, x in enumerate(X[:15]):
        print("".join([i_to_char[y] for y in x]), "→", i_to_char[Y[i]])

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return (X,)


@app.cell
def _(char_to_i, torch):
    # Embedding matrix (maps 27 chars -> 2D space)
    C = torch.randn([len(char_to_i), 2])
    return (C,)


@app.cell
def _(C, X):
    X_embedded = C[X]
    return


if __name__ == "__main__":
    app.run()
