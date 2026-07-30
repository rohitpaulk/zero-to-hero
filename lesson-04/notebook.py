import marimo

__generated_with = "0.23.15"
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

    return F, Path, plt, torch


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

    for word in words:
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

    dataset = torch.utils.data.TensorDataset(X, Y)

    train_dataset, dev_dataset, test_dataset = torch.utils.data.dataset.random_split(
        dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(1)
    )

    X_train, Y_train = train_dataset[:]
    X_dev, Y_dev = dev_dataset[:]
    X_test, Y_test = test_dataset[:]
    return X_test, X_train, Y_test, Y_train, block_size


@app.cell
def _(block_size, char_to_i, torch):
    # Generator (for reproducibility)
    g = torch.Generator().manual_seed(1)

    # Model dimensions
    embedding_dimensions = 8
    hidden_dimensions = 300
    vocab_size = len(char_to_i)

    # Embedding matrix
    C = torch.randn((vocab_size, embedding_dimensions), generator=g)

    # Hidden layer
    W1 = torch.empty((block_size * embedding_dimensions, hidden_dimensions))
    torch.nn.init.xavier_normal_(W1, gain=torch.nn.init.calculate_gain("tanh"), generator=g)
    b1 = torch.zeros(hidden_dimensions)

    # Output layer
    W2 = torch.empty((hidden_dimensions, vocab_size))
    torch.nn.init.xavier_normal_(W2, generator=g)
    b2 = torch.zeros(vocab_size)

    parameters = [C, W1, b1, W2, b2]

    def _():
        for p in parameters:
            p.requires_grad = True

    _()
    return C, W1, W2, b1, b2, embedding_dimensions, parameters


@app.cell
def _(
    C,
    F,
    W1,
    W2,
    X_test,
    X_train,
    Y_test,
    Y_train,
    b1,
    b2,
    block_size,
    embedding_dimensions,
    torch,
):
    def _calculate_loss(x, y):
        x_embedded = C[x]
        h = torch.tanh(x_embedded.view(-1, block_size * embedding_dimensions) @ W1 + b1)
        logits = h @ W2 + b2

        loss = F.cross_entropy(logits, y)
        return loss.item()

    def calculate_training_loss():
        return _calculate_loss(X_train, Y_train)

    def calculate_test_loss():
        return _calculate_loss(X_test, Y_test)

    return calculate_test_loss, calculate_training_loss


@app.cell
def _(
    C,
    F,
    W1,
    W2,
    X_train,
    Y_train,
    b1,
    b2,
    block_size,
    calculate_training_loss,
    embedding_dimensions,
    parameters,
    plt,
    torch,
):
    batch_size = 64

    losses = []

    for loop_index in range(50000):
        batch_indices = torch.randint(0, len(X_train), (batch_size,))

        X_batch = X_train[batch_indices]
        Y_batch = Y_train[batch_indices]

        # Forward pass
        X_embedded = C[X_batch]
        h = torch.tanh(X_embedded.view(-1, block_size * embedding_dimensions) @ W1 + b1)
        logits = h @ W2 + b2

        loss = F.cross_entropy(logits, Y_batch)

        for p in parameters:
            p.grad = None

        loss.backward()

        for p in parameters:
            p.data -= 0.01 * p.grad

        if loop_index % 1000 == 0:
            training_loss = calculate_training_loss()
            losses.append(training_loss)
            print(training_loss)


    plt.plot(losses)
    return


@app.cell
def _(calculate_test_loss):
    print(calculate_test_loss())
    return


if __name__ == "__main__":
    app.run()
