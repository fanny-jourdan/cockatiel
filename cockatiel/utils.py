import torch
from math import ceil
import numpy as np

from flair.models import SequenceTagger
from flair.data import Sentence

from typing import List, Callable, Union, Optional, Tuple


# load tagger
tagger = SequenceTagger.load("flair/chunk-english")


def batcher(elements, batch_size: int):
    """
    An iterable to create batches from a list of elements
    """
    nb_batchs = ceil(len(elements) / batch_size)

    for batch_i in range(nb_batchs):
        batch_start = batch_i * batch_size
        batch_end = batch_start + batch_size

        batch = elements[batch_start:batch_end]
        yield batch


def tokenize(samples: List[str], tokenizer: Callable, device='cuda'):
    """
    A function to transform a list of strings into tokens to be consumed by the transformer model.
    """
    x = tokenizer(
        [s for s in samples],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors='pt'
    ).to(device)

    return x


def preprocess(samples: List[Tuple[str, str]], tokenizer: Callable, device='cuda'):
    """
    A basic pre-processing function to transform the format from the imdb dataset to
    something easier to work with.
    """
    x, y = samples[:, 0], samples[:, 1]
    x = tokenize(x, tokenizer, device)
    y = torch.Tensor(y == 'positive').int().to(device)
    return x, y


def batch_predict(model, tokenizer: Callable, inputs: List[Tuple[str, str]], batch_size: int = 64, device='cuda'):
    """
    A function to pre-process and predict using the transformer model in batches.
    """
    predictions = None
    labels = None

    with torch.no_grad():
        for batch_input in batcher(inputs, batch_size):
            xp, yp = preprocess(batch_input, tokenizer, device)
            out_batch = model(**xp)
            predictions = out_batch if predictions is None else torch.cat([predictions, out_batch])
            labels = yp if labels is None else torch.cat([labels, yp])

        return predictions, labels


def extract_clauses(ds_entry: Union[List[str], str], clause_type=['NP', 'ADJP']) -> List[str]:
    """
    Separates the input texts into clauses, and only keeps the ones belonging to the specified types.
    If clause_type is None, the texts are split but all the clauses are kept.

    Parameters
    ----------
    ds_entry
        A list of strings that we wish to separate into clauses.
    clause_type
        A list with the types of clauses to keep. If None, all clauses are kept.

    Returns
    -------
    clause_list
        A list with input texts split into clauses.
    """
    s = Sentence(ds_entry)
    tagger.predict(s)
    clause_list = []
    for segment in s.get_labels():
        if clause_type is None:
            clause_list.append(segment.data_point.text)
        elif segment.value in clause_type:
            clause_list.append(segment.data_point.text)

    return clause_list


def batch_activations_fct(model, inputs: List[str], batch_size=64) -> torch.Tensor:
    """
    A function to extract the activations of input texts in batches.
    """
    activations = None
    with torch.no_grad():
        for batch_input in batcher(inputs, batch_size):
            out_batch = model.features(**batch_input)
            activations = out_batch if activations is None else torch.cat([activations, out_batch])
        return activations


def acti_preprocess(activations: torch.Tensor) -> np.ndarray:
    """
    A function to preprocess the activations to work with COCKATIEL
    """
    if len(activations.shape) == 4:
        activations = torch.mean(activations, (1, 2))

    if isinstance(activations, np.ndarray):
        activations = torch.Tensor(activations)
    if torch.min(activations) < 0:
        raise ValueError("Please choose a layer with positive activations.")

    return activations.cpu().numpy().astype(np.float32)


def calculate_u_values(sentence, cropped_sentences, model, tokenizer, factorization,
                       separate, ignore_words: Optional[List[str]] = None, device='cuda') -> np.ndarray:
    if ignore_words is None:
        ignore_words = []
    with torch.no_grad():
        activations = None
        for crop_id in range(-1, len(cropped_sentences)):
            if crop_id == -1:
                perturbated_review = sentence
            elif cropped_sentences[crop_id] not in ignore_words:
                perturbated_review = separate.join(np.delete(cropped_sentences, crop_id))
            else:
                continue
            tokenized_perturbated_review = tokenizer(perturbated_review, truncation=True, padding=True, return_tensors="pt").to(device)
            activation = model.features(**tokenized_perturbated_review)
            activations = activation if activations is None else torch.cat([activations, activation])

        activations = acti_preprocess(activations)
        u_values = factorization.transform(activations)
        return u_values


def calculate_importance(
        words: List[str], u_values: np.ndarray, concept_id: int, ignore_words: List[str]
) -> List[float]:
    """
    Calculates the presence of concepts in the input list of words.
    """
    u_delta = u_values[0, concept_id] - u_values[1:, concept_id]
    importances = []
    delta_id = 0  # pointer to get current id in importance (as we skip unused word)

    for word_id in range(len(words)):
        if words[word_id] not in ignore_words:
            importances.append(u_delta[delta_id])
            delta_id += 1
        else:
            importances.append(0.0)

    return importances
