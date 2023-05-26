import torch
from math import ceil


def batcher(elements, batch_size: int):
    nb_batchs = ceil(len(elements) / batch_size)

    for batch_i in range(nb_batchs):
        batch_start = batch_i * batch_size
        batch_end = batch_start + batch_size

        batch = elements[batch_start:batch_end]
        yield batch


def tokenize(samples, tokenizer, device='cuda'):
    x = tokenizer([s for s in samples], padding="max_length", max_length = 512, truncation=True, return_tensors='pt').to(device)
    return x


def preprocess(samples, tokenizer, device='cuda'):
    x, y = samples[:, 0], samples[:, 1]
    x = tokenize(x, tokenizer, device)
    if device == 'cuda':
        y = torch.Tensor(y == 'positive').int().cuda()
    else:
        y = torch.Tensor(y == 'positive').int()
    return x, y


def batch_predict(model, tokenizer, inputs, batch_size: int = 64, device='cuda'):
    predictions = None
    labels = None

    with torch.no_grad():
        for batch_input in batcher(inputs, batch_size):
            xp, yp = preprocess(batch_input, tokenizer, device)
            out_batch = model(**xp)
            predictions = out_batch if predictions is None else torch.cat([predictions, out_batch])
            labels = yp if labels is None else torch.cat([labels, yp])

        return predictions, labels
