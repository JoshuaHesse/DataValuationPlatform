#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:43:05 2023

@author: joshua

This file contains custom tanimoto kernels used for sklearn SVR, as well as
sklearn GPR and gpflow GPR. 

"""
import numpy as np
import gpflow
from gpflow.utilities import positive
from gpflow.utilities.ops import broadcasting_elementwise
import tensorflow as tf
from sklearn.gaussian_process.kernels import Kernel
    
class TanimotoSVR:
    """
    This class represents a custom Tanimoto Kernel to be used as a custom Kernel for Sklearn SVRs. 
    It's based on the Tanimoto Kernel implementation found on the provided URL and has been adapted 
    to suit the specific use case.
    (https://anonymous.4open.science/r/f160a0a2-0161-4d31-ba55-2e3aab2133d3/GP/kernels.py)
    
    Attributes
    ----------
    variance : int, default=1.0
        A scalar factor that scales output of the Tanimoto kernel. Default value is 1. 

    Methods
    -------
    __call__(self, X, Y=None, eval_gradient=False)
        Computes Tanimoto kernel between two sets of vectors.
        
        Parameters
        ----------
        X : np.array
            A MxD array where M is the number of samples and D is the number of features.
        Y : np.array, optional
            Another MxD array where M is the number of samples and D is the number of features. 
            If not provided, assumed to be equal to X.
        eval_gradient: bool, optional
            Whether to compute derivative of Tanimoto kernel with respect to its parameters. 
            This is not implemented at the moment.
        
        Returns
        -------
        K : np.array
            The MxM matrix of all pairwise Tanimoto similarities between the samples in X, and Y 
            if Y is provided.
    """
    def __init__(self, variance: int = 1.0):
        self.variance = variance

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        Xs = np.sum(np.square(X), axis=-1)[:, np.newaxis]
        Ys = np.sum(np.square(Y), axis=-1)[np.newaxis, :]
        outer_product = X @ Y.T
        denominator = -outer_product + Xs + Ys

        K = self.variance * outer_product / denominator

        if eval_gradient:
            raise NotImplementedError("Gradient computation is not supported.")

        return K


#-----------------------------------------------------------------------------#


class Tanimoto(gpflow.kernels.Kernel):
    """
    This class represents a custom Tanimoto kernel based on the gpflow kernel, which is used 
    for Gaussian Process Regression on high dimensional input, like ecfp. It inherits from the 
    gpflow Kernel base class. The original code was obtained from the provided URL.
    https://anonymous.4open.science/r/f160a0a2-0161-4d31-ba55-2e3aab2133d3/GP/kernels.py
    
    Attributes
    ----------
    variance: gpflow.Parameter object
        A gpflow.Parameter object storing a positive scalar that scales 
        the output of the Tanimoto kernel. It gets auto initialized to 1.0 and
        is constrained to be positive during optimization.
    Methods
    -------
    K(self, X, X2=None) -> tf.Tensor
        Computes the Tanimoto kernel matrix. If X2 is not provided, 
        it computes the NxN kernel matrix for X.
        
    K_diag(self, X) -> tf.Tensor
        Computes the diagonal of the NxN kernel matrix of X.
        
    """
    def __init__(self):
        super().__init__()
        # We constrain the value of the kernel variance to be positive when it's being optimised
        self.variance = gpflow.Parameter(1.0, transform=positive())

    def K(self, X, X2=None):
        """
        Compute the Tanimoto kernel matrix σ² * ((<x, y>) / (||x||^2 + ||y||^2 - <x, y>))
        :param X: N x D array
        :param X2: M x D array. If None, compute the N x N kernel matrix for X.
        :return: The kernel matrix of dimension N x M
        """
        if X2 is None:
            X2 = X

        Xs = tf.reduce_sum(tf.square(X), axis=-1)  # Squared L2-norm of X
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)  # Squared L2-norm of X2
        outer_product = tf.tensordot(X, X2, [[-1], [-1]])  # outer product of the matrices X and X2

        # Analogue of denominator in Tanimoto formula

        denominator = -outer_product + broadcasting_elementwise(tf.add, Xs, X2s)

        return self.variance * outer_product/denominator

    def K_diag(self, X):
        """
        Compute the diagonal of the N x N kernel matrix of X
        :param X: N x D array
        :return: N x 1 array
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

#-----------------------------------------------------------------------------#

class TanimotoSK(Kernel):
    """
    
   This class represents a custom Tanimoto kernel, originally based on the gpflow kernel, 
   used for Gaussian Process Regression on high dimensional input, such as ecfp. It inherits 
   from the Scikit-Learn Kernel base class. The original code was adapted from the provided URL 
   to work with the Scikit-Learn's GPR implementation.
   https://anonymous.4open.science/r/f160a0a2-0161-4d31-ba55-2e3aab2133d3/GP/kernels.py
   
   Attributes
   ----------
   variance: float
       A scalar that scales the output of the Tanimoto kernel. It gets auto initialized to 1.0.
       
   Methods
   -------
   __call__(self, X, Y=None, eval_gradient=False) -> np.ndarray
       Computes the Tanimoto kernel matrix K of X and Y. If Y is not 
       provided, it computes the NxN kernel matrix for X. It does not support 
       gradient computation.
       
   diag(self, X) -> np.ndarray
       Returns array filled with self.variance
        
   is_stationary(self) -> bool
       Returns False: The kernel is not stationary.
       
   
    and then adapted to work for Sklearns GPR implementation
    
    """
    
    def __init__(self, variance=1.0):
        self.variance = variance

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Y = X

        Xs = np.sum(np.square(X), axis=-1)[:, np.newaxis]
        Ys = np.sum(np.square(Y), axis=-1)[np.newaxis, :]
        outer_product = X @ Y.T
        denominator = -outer_product + Xs + Ys

        K = self.variance * outer_product / denominator

        if eval_gradient:
            raise NotImplementedError("Gradient computation is not supported.")

        return K

    def diag(self, X):
        return np.full(X.shape[0], self.variance)

    def is_stationary(self):
        return False



 