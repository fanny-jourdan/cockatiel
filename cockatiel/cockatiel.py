import numpy as np
import torch
from typing import Dict, Tuple
from sklearn.decomposition import NMF
from math import ceil
from .sampling import ScipySobolSequence, concept_perturbation
from .sobol import JansenEstimator
from .model_fct import tokenize


class COCKATIEL:
    sobol_nb_design = 32

    def __init__(self, model, tokenizer, components: int = 25, batch_size: int = 256, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.components = components
        self.batch_size = batch_size
        self.device = device

    def extract_concepts(self, cropped_dataset, dataset, class_id: int, limit_sobol: int = 1_000_000):
        segments = []
        segments_activations = None

        with torch.no_grad():
            for batch_id in range(ceil(len(cropped_dataset) / self.batch_size)):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size

                batch_sentences = cropped_dataset[batch_start:batch_end]
                batch_tokenized = tokenize(batch_sentences, self.tokenizer, self.device)

                batch_activations = self.model.features(**batch_tokenized)

                segments_activations = batch_activations if segments_activations is None \
                    else torch.cat([segments_activations, batch_activations], 0)
                segments = batch_sentences if segments is None \
                    else segments + batch_sentences

        points_activations = None
        with torch.no_grad():
            for batch_id in range(ceil(len(dataset) / self.batch_size)):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size
                tokenized_batch = tokenize(dataset[batch_start:batch_end], self.tokenizer, self.device)
                activations = self.model.features(**tokenized_batch)
                points_activations = activations if points_activations is None else torch.cat(
                    [points_activations, activations])

        # applying GAP(.) on the activation and ensure positivity if needed
        segments_activations = self._preprocess(segments_activations)
        points_activations = self._preprocess(points_activations)

        # using the activations, we will now use the matrix factorization to
        # find the concept bank (W) and the concept representation (U) of the
        # segments and the points
        factorization = NMF(n_components=self.components)

        u_segments = factorization.fit_transform(segments_activations)
        u_points = factorization.transform(points_activations)

        if self.device == 'cuda':
            W = torch.Tensor(factorization.components_).float().cuda()
        else:
            W = torch.Tensor(factorization.components_).float()

        # we don't need segments activations anymore, the concept bank is trained
        del segments_activations

        # using the concept bank and the points, we will now evaluate the importance of
        # each concept for each points to get a global importance score for each
        # concept in the concept bank
        global_importance = self._sobol_importance(cropped_dataset, points_activations[:limit_sobol], class_id, W)

        return segments, u_segments, factorization, global_importance

    def _sobol_importance(self, cropped_dataset, activations, class_id, W):
        masks = ScipySobolSequence()(self.components, nb_design=self.sobol_nb_design)
        estimator = JansenEstimator()

        if self.device == 'cuda':
            W = torch.Tensor(W).float().cuda()
        else:
            W = torch.Tensor(W).float()

        importances = []
        for act in activations:
            act = torch.Tensor(act).float()
            if self.device == 'cuda':
                act = act.cuda()

            y_pred = None
            for batch_id in range(ceil(len(cropped_dataset) / self.batch_size)):
                batch_start = batch_id * self.batch_size
                batch_end = batch_start + self.batch_size

                if self.device == 'cuda':
                    batch_masks = torch.Tensor(masks[batch_start:batch_end]).float().cuda()
                else:
                    batch_masks = torch.Tensor(masks[batch_start:batch_end]).float()

                y_batch = concept_perturbation(self.model, act, batch_masks, class_id, W)
                y_pred = y_batch if y_pred is None else torch.cat([y_pred, y_batch], 0)

            if self.device == 'cuda':
                y_pred = y_pred.cpu()
            stis = estimator(masks, y_pred.numpy(), self.sobol_nb_design)
            importances.append(stis)

        global_importance = np.mean(importances, 0)

        return global_importance

    def _preprocess(self, activations):
        if len(activations.shape) == 4:
            activations = torch.mean(activations, (1, 2))

        if torch.min(activations) < 0:
            raise ValueError("Please choose a layer with positive activations.")
        if self.device == 'cuda':
            activations = activations.cpu()

        return activations.numpy().astype(np.float32)
