"""
Sobol' total order estimators module
"""

from abc import ABC, abstractmethod
import numpy as np


class SobolEstimator(ABC):
    """
    Base class for Sobol' total order estimators.
    """

    @staticmethod
    def masks_dim(masks):
        """
        Deduce the number of dimensions using the sampling masks.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        nb_dim
          The number of dimensions under study according to the masks.
        """
        nb_dim = np.prod(masks.shape[1:])
        return nb_dim

    @staticmethod
    def split_abc(outputs, nb_design, nb_dim):
        """
        Split the outputs values into the 3 sampling matrices A, B and C.

        Parameters
        ----------
        outputs
          Model outputs for each sample point of matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).
        nb_dim
          Number of dimensions to estimate.

        Returns
        -------
        a
          The results for the sample points in matrix A.
        b
          The results for the sample points in matrix B.
        c
          The results for the sample points in matrix C.
        """
        sampling_a = outputs[:nb_design]
        sampling_b = outputs[nb_design:nb_design*2]
        replication_c = np.array([outputs[nb_design*2 + nb_design*i:nb_design*2 + nb_design*(i+1)]
                      for i in range(nb_dim)])
        return sampling_a, sampling_b, replication_c

    @staticmethod
    def post_process(stis, masks):
        """
        Post processing ops on the indices before sending them back. Makes sure the data
        format and shape is correct.

        Parameters
        ----------
        stis
          Total order Sobol' indices, one for each dimensions.
        masks
            Low resolution masks (before upsampling) used, one for each output.

        Returns
        -------
        stis
          Total order Sobol' indices after post processing.
        """
        stis = np.array(stis, np.float32)
        return stis.reshape(masks.shape[1:])

    @abstractmethod
    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Ref. Jansen, M., Analysis of variance designs for model output (1999)
        https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti: ndarray
          Total order Sobol' indices, one for each dimensions.
        """
        raise NotImplementedError()


class JansenEstimator(SobolEstimator):
    """
    Jansen estimator for total order Sobol' indices.

    Ref. Jansen, M., Analysis of variance designs for model output (1999)
    https://www.sciencedirect.com/science/article/abs/pii/S0010465598001544
    """

    def __call__(self, masks, outputs, nb_design):
        """
        Compute the Sobol' total order indices according to the Jansen algorithm.

        Parameters
        ----------
        masks
          Low resolution masks (before upsampling) used, one for each output.
        outputs
          Model outputs associated to each masks. One for each sample point of
          matrices A, B and C (in order).
        nb_design
          Number of points for matrices A (the same as B).

        Returns
        -------
        sti
          Total order Sobol' indices, one for each dimensions.
        """
        nb_dim = self.masks_dim(masks)
        sampling_a, _, replication_c = self.split_abc(outputs, nb_design, nb_dim)

        mu_a = np.mean(sampling_a)
        var = np.sum([(v - mu_a)**2 for v in sampling_a]) / (len(sampling_a) - 1)

        stis = [
            np.sum((sampling_a - replication_c[i])**2.0) / (2 * nb_design * var)
            for i in range(nb_dim)
        ]

        return self.post_process(stis, masks)
    