"""
Sampling methods for replicated designs
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy


class Sampler(ABC):
    """
    Base class for replicated design sampling.
    """

    @staticmethod
    def build_replicated_design(sampling_a, sampling_b):
        """
        Build the replicated design matrix C using A & B

        Parameters
        ----------
        sampling_a
          The masks values for the sampling matrix A.
        sampling_b
          The masks values for the sampling matrix B.

        Returns
        -------
        replication_c
          The new replicated design matrix C generated from A & B.
        """
        replication_c = np.array([sampling_a.copy() for _ in range(sampling_a.shape[-1])])
        for i in range(len(replication_c)):
            replication_c[i, :, i] = sampling_b[:, i]

        replication_c = replication_c.reshape((-1, sampling_a.shape[-1]))

        return replication_c

    @abstractmethod
    def __call__(self, dimension, nb_design):
        raise NotImplementedError()


class ScipySampler(Sampler):
    """
    Base class based on Scipy qmc module for replicated design sampling.
    """

    def __init__(self):
        try:
            self.qmc = scipy.stats.qmc
        except AttributeError as err:
            raise ModuleNotFoundError("COCKATIEL need scipy>=1.7 to use this sampling.") from err


class ScipySobolSequence(ScipySampler):
    """
    Scipy Sobol LP tau sequence sampler.

    Ref. I. M. Sobol., The distribution of points in a cube and the accurate evaluation of
    integrals (1967).
    https://www.sciencedirect.com/science/article/abs/pii/0041555367901449
    """

    def __call__(self, dimension, nb_design):
        sampler = self.qmc.Sobol(dimension * 2, scramble=False)
        sampling_ab = sampler.random(nb_design).astype(np.float32)
        sampling_a, sampling_b = sampling_ab[:, :dimension], sampling_ab[:, dimension:]
        replicated_c = self.build_replicated_design(sampling_a, sampling_b)

        return np.concatenate([sampling_a, sampling_b, replicated_c], 0)


def concept_perturbation(model, activation, masks, class_id, W):
    """
    Apply perturbation on the concept before reconstruction and get the perturbated outputs.
    For NMF we recall that A = U @ W

    Parameters
    ----------
    model
      Model that map the concept layer to the output (h_l->k in the paper)
    activation
      Specific activation to apply perturbation on.
    masks
      Arrays of masks, each of them being a concept perturbation.
    class_id
      Id the class to test.
    W
      Concept bank extracted using NMF.

    Returns
    -------
    y
      Outputs of the perturbated points.
    """
    perturbation = masks @ W

    if len(activation.shape) == 3:
        perturbation = perturbation[:, None, None, :]

    activation = activation[None, :]
    perturbated_activations = activation + perturbation * activation
    y = model.end_model(perturbated_activations)[:, class_id]

    return y
