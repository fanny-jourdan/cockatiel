"""
This module implements the local part of COCKATIEL: occlusion. This allows us to estimate the presence of
concepts in parts of the input text.
"""

import numpy as np
import nltk
import sklearn.decomposition
from nltk.tokenize import word_tokenize

from .utils import extract_clauses, calculate_u_values, calculate_importance

from typing import List, Callable, Optional, Union, Tuple

nltk.download('punkt')


def occlusion_concepts(
        sentence: str,
        model,
        tokenizer: Callable,
        factorization: Union[sklearn.decomposition.NMF, Tuple[sklearn.decomposition.NMF, sklearn.decomposition.NMF]],
        l_concept_id: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        ignore_words: Optional[List[str]] = None,
        two_labels: bool = True,
        extract_fct: str = "clause",
        device='cuda'
) -> np.ndarray:
    """
    Generates explanations for the input sentence using COCKATIEL.

    If two_labels is False, it computes the presence of the concepts of interest (in l_concept_id) using the
    NMF object in factorization.
    If two_labels is True, it computes the presence of the concepts of interest in the tuple of l_concept_id using
    the tuple of NMF objects in factorization (to do so for both classes in imdb-reviews task).

    The granularity of the explanations is set with extract_fct.

    Parameters
    ----------
    sentence
        The string (sentence) we wish to explain using COCKATIEL.
    model
        The model under study.
    tokenizer
        A Callable that transforms strings into tokens capable of being ingested by the model.
    factorization
        Either the NMF object to transform activations into the concept base, or a tuple with an object for each of
        the two classes (for the imdb-reviews task) if two_labels is True.
    l_concept_id
        Either a list of concepts of interest (for a given task) or a tuple with two lists, one for each class (if
        two_labels is True).
    ignore_words
        A list of strings to ignore when applying occlusion.
    two_labels
        A bool indicating whether we wish to explain only one class or both (for imdb-reviews task).
    extract_fct
        A string indicating whether at which level we wish to explain: "word", "clause" or "sentence".
    device
        The device on which tensors are stored ("cpu" or "cuda").

    Returns
    -------
    l_importances
        An array with the presence of each concept in the input sentence.
    """
    sentence = str(sentence)

    if extract_fct == "clause":
        words = extract_clauses(sentence, clause_type=None)
        separate = " "

    else:
        words = word_tokenize(sentence)
        if extract_fct == "sentence":
            separate = ". "
        elif extract_fct == "word":
            separate = " "
        else:
            raise ValueError("Extraction function can be only 'clause', 'sentence', or 'word")

    if two_labels:
        u_values_pos = calculate_u_values(sentence, words,  model, tokenizer, factorization[0], separate, ignore_words, device)
        u_values_neg = calculate_u_values(sentence, words,  model, tokenizer, factorization[1], separate, ignore_words, device)
        l_importances = []
        for concept_id in l_concept_id[0]:
            importances = calculate_importance(words, u_values_pos, concept_id, ignore_words)
            l_importances.append(np.array(importances))
        for concept_id in l_concept_id[1]:
            importances = calculate_importance(words, u_values_neg, concept_id, ignore_words)
            l_importances.append(np.array(importances))

    else:  # look at only one class:
        u_values = calculate_u_values(sentence, words,  model, tokenizer, factorization, separate, ignore_words, device)
        l_importances = []
        for concept_id in l_concept_id:
            importances = calculate_importance(words, u_values, concept_id, ignore_words)
            l_importances.append(np.array(importances))

    return np.array(l_importances)
