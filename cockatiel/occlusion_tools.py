from math import ceil
import numpy as np
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from .model_fct import batcher


# load tagger
tagger = SequenceTagger.load("flair/chunk-english")


def extract_clauses(ds_entry, clause_type=['NP', 'ADJP']):
    """
    ds_entry, a string
    clause_type, a list with the type of clause we can keep.
    To have all the clauses, put clause_type = None.
    """
    s = Sentence(ds_entry)
    tagger.predict(s)
    segments = []
    for segment in s.get_labels():
        if clause_type is None:
            segments.append(segment.data_point.text)
        elif segment.value in clause_type:
            segments.append(segment.data_point.text)
    return segments


def batch_activations_fct(model, inputs, batch_size=64):
    activations = None
    with torch.no_grad():
        for batch_input in batcher(inputs, batch_size):
            out_batch = model.features(**batch_input)
            activations = out_batch if activations is None else torch.cat([activations, out_batch])
        return activations


def acti_preprocess(activations):
    if len(activations.shape) == 4:
        activations = torch.mean(activations, (1, 2))

    if isinstance(activations, np.ndarray):
        activations = torch.Tensor(activations)
    if torch.min(activations) < 0:
        raise ValueError("Please choose a layer with positive activations.")

    return activations.cpu().numpy().astype(np.float32)


def calculate_u_values(sentence, cropped_sentences, model, tokenizer, factorization, separate, ignore_words = [], device = 'cuda'):
    with torch.no_grad():
        activations = None
        for crop_id in range(-1, len(cropped_sentences)):
            if crop_id == -1:
                perturbated_review = sentence
            elif cropped_sentences[crop_id] not in ignore_words:
                perturbated_review = separate.join(np.delete(cropped_sentences, crop_id))
            else:
                continue
            tokenized_perturbated_review = tokenizer(perturbated_review, truncation=True, padding=True, return_tensors="pt")
            if device == 'cuda':
                tokenized_perturbated_review = tokenized_perturbated_review.to('cuda')
            activation = model.features(**tokenized_perturbated_review)
            activations = activation if activations is None else torch.cat([activations, activation])

        activations = acti_preprocess(activations)
        u_values = factorization.transform(activations)
        return u_values


def calculate_importance(words, u_values, concept_id, ignore_words):
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
