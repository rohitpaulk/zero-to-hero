import marimo

__generated_with = "0.23.16"
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
    block_size = 3
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
    return X_train, Y_train, block_size


@app.cell
def _(torch):
    def compare(label, calculated_grad, tensor):
        actual_grad = tensor.grad

        if calculated_grad.shape != actual_grad.shape:
            print(f"Calculated shape: {calculated_grad.shape}, expected: {actual_grad.shape}")
            return

        is_exact_match = torch.equal(calculated_grad, actual_grad)
        is_close_match = torch.allclose(calculated_grad, actual_grad)
        max_absolute_difference = (calculated_grad - actual_grad).abs().max().item()

        print(
            f"{label:15s} | "
            f"exact: {str(is_exact_match):5s} | "
            f"approximate: {str(is_close_match):5s} | "
            f"maxdiff: {max_absolute_difference}"
        )

    return (compare,)


@app.cell
def _(block_size, char_to_i, torch):
    # Generator (for reproducibility)
    g = torch.Generator().manual_seed(2147483647)

    # Model dimensions
    embedding_dimensions = 10
    hidden_dimensions = 64
    vocab_size = len(char_to_i)

    # Embedding matrix
    C = torch.randn((vocab_size, embedding_dimensions), generator=g)

    # Hidden layer
    W1 = (
        torch.randn((block_size * embedding_dimensions, hidden_dimensions), generator=g)
        * (5 / 3)
        / ((embedding_dimensions * block_size) ** 0.5)
    )
    b1 = torch.randn(hidden_dimensions, generator=g) * 0.1

    # Output layer
    W2 = torch.randn((hidden_dimensions, vocab_size), generator=g) * 0.1
    b2 = torch.randn(vocab_size, generator=g) * 0.1

    batch_gain = torch.randn((1, hidden_dimensions)) * 0.1 + 1.0
    batch_bias = torch.randn((1, hidden_dimensions)) * 0.1

    parameters = [C, W1, b1, W2, b2, batch_gain, batch_bias]
    print(sum(p.nelement() for p in parameters))

    def _():
        for p in parameters:
            p.requires_grad = True

    _()
    return C, W1, W2, b1, b2, batch_bias, batch_gain, g, parameters


@app.cell
def _(X_train, Y_train, g, torch):
    batch_size = 32

    # construct a minibatch
    batch_indices = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)
    X_batch = X_train[batch_indices]
    Y_batch = Y_train[batch_indices]
    return X_batch, Y_batch, batch_size


@app.cell
def _(
    C,
    W1,
    W2,
    X_batch,
    Y_batch,
    b1,
    b2,
    batch_bias,
    batch_gain,
    batch_size,
    parameters,
    torch,
):
    # forward pass, "chunkated" into smaller steps that are possible to backward one at a time

    X_embedded = C[X_batch]  # embed the characters into vectors
    X_embedded_cat = X_embedded.view(X_embedded.shape[0], -1)  # concatenate the vectors

    # Linear layer 1
    h_pre_norm = X_embedded_cat @ W1 + b1  # hidden layer pre-activation

    # BatchNorm layer
    batch_mean_i = 1 / batch_size * h_pre_norm.sum(0, keepdim=True)
    batch_diff = h_pre_norm - batch_mean_i
    batch_diff_sq = batch_diff**2
    batch_var = (
        1 / (batch_size - 1) * (batch_diff_sq).sum(0, keepdim=True)
    )  # note: Bessel's correction (dividing by n-1, not n)
    batch_var_inv = (batch_var + 1e-5) ** -0.5
    batch_raw = batch_diff * batch_var_inv
    h_pre_activation = (batch_raw * batch_gain) + batch_bias

    # Non-linearity
    h = torch.tanh(h_pre_activation)  # hidden layer

    # Linear layer 2
    logits = h @ W2 + b2  # output layer

    # cross entropy loss (same as F.cross_entropy(logits, Yb))
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes  # subtract max for numerical stability
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdims=True)
    counts_sum_inv = counts_sum**-1  # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(batch_size), Y_batch].mean()

    # PyTorch backward pass
    for p in parameters:
        p.grad = None

    for t in [
        logprobs,
        probs,
        counts,
        counts_sum,
        counts_sum_inv,  # afaik there is no cleaner way
        norm_logits,
        logit_maxes,
        logits,
        h,
        h_pre_activation,
        batch_raw,
        batch_var_inv,
        batch_var,
        batch_diff,
        batch_diff_sq,
        h_pre_norm,
        batch_mean_i,
        X_embedded_cat,
        X_embedded,
    ]:
        t.retain_grad()

    loss.backward()
    loss
    return (
        batch_diff,
        batch_raw,
        batch_var_inv,
        counts,
        counts_sum,
        counts_sum_inv,
        h,
        h_pre_activation,
        logit_maxes,
        logits,
        logprobs,
        norm_logits,
        probs,
    )


@app.cell
def _(
    W2,
    Y_batch,
    b2,
    batch_bias,
    batch_diff,
    batch_gain,
    batch_raw,
    batch_size,
    batch_var_inv,
    compare,
    counts,
    counts_sum,
    counts_sum_inv,
    h,
    h_pre_activation,
    logit_maxes,
    logits,
    logprobs,
    norm_logits,
    probs,
    torch,
):
    # loss = -logprobs[range(batch_size), Y_batch].mean()
    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[range(batch_size), Y_batch] = -1 / batch_size
    compare("logprobs", dlogprobs, logprobs)

    # logprobs = probs.log()
    dprobs = dlogprobs * 1 / probs
    compare("probs", dprobs, probs)

    dprobs.shape

    # probs = counts * counts_sum_inv
    dcounts_sum_inv = (dprobs * counts).sum(1, keepdims=True)
    dcounts = dprobs * counts_sum_inv
    compare("counts_sum_inv", dcounts_sum_inv, counts_sum_inv)

    # counts_sum_inv = counts_sum**-1
    dcounts_sum = dcounts_sum_inv * (-(counts_sum**-2))
    compare("counts_sum", dcounts_sum, counts_sum)

    # counts_sum = counts.sum(1, keepdims=True)
    dcounts += dcounts_sum * torch.ones_like(counts)
    compare("counts", dcounts, counts)

    # counts = norm_logits.exp()
    dnorm_logits = dcounts * norm_logits.exp()
    compare("norm_logits", dnorm_logits, norm_logits)

    # norm_logits = logits - logit_maxes
    dlogit_maxes = -dnorm_logits.sum(1, keepdims=True)
    dlogits = dnorm_logits
    compare("logit_maxes", dlogit_maxes, logit_maxes)

    # logit_maxes = logits.max(1, keepdim=True).values
    # logit_maxes.shape = 32 x 1 (1 item per batch entry)
    # logits.shape = 32 x 27 (27 items per batch entry)
    # dlogits = 32 x 27
    max_indices = logits.argmax(dim=1, keepdim=True)
    dlogits.scatter_add_(dim=1, index=max_indices, src=dlogit_maxes)
    compare("logits", dlogits, logits)

    # logits =   h   @   W2  +  b2
    dh = dlogits @ W2.transpose(0, 1)
    dW2 = h.transpose(0, 1) @ dlogits
    db2 = dlogits.sum(axis=0)
    compare('h', dh, h)
    compare('W2', dW2, W2)
    compare('b2', db2, b2)

    # h = torch.tanh(h_pre_activation)  # hidden layer
    dh_pre_activation = dh * (1 - h**2)
    compare('h_pre_activate', dh_pre_activation, h_pre_activation)

    # h_pre_activation = (batch_raw * batch_gain) + batch_bias
    dbatch_gain = (dh_pre_activation * batch_raw).sum(axis=0, keepdim=True)
    dbatch_raw = (dh_pre_activation * batch_gain)
    dbatch_bias = dh_pre_activation.sum(axis=0, keepdim=True)
    compare('batch_gain', dbatch_gain, batch_gain)
    compare('batch_raw', dbatch_raw, batch_raw)
    compare('batch_bias', dbatch_bias, batch_bias)

    # batch_raw = batch_diff * batch_var_inv
    dbatch_var_inv = (dbatch_raw * batch_diff).sum(axis=0, keepdim=True)
    print(batch_raw.shape, batch_diff.shape, batch_var_inv.shape)
    dbatch_diff = dbatch_raw * batch_var_inv
    compare('dbatch_var_inv', dbatch_var_inv, batch_var_inv)

    # compare('bnvar', dbnvar, bnvar)
    # compare('bndiff2', dbndiff2, bndiff2)
    # compare('dbatch_diff', dbatch_diff, batch_diff)
    # compare('bnmeani', dbnmeani, bnmeani)
    # compare('hprebn', dhprebn, hprebn)
    # compare('embcat', dembcat, embcat)
    # compare('W1', dW1, W1)
    # compare('b1', db1, b1)
    # compare('emb', demb, emb)
    # compare('C', dC, C)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
