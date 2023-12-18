#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:49:21 2023

@author: joshua

This file contains the sampling functions used in the active_learning_pipeline.
"""
import numpy as np



def ucb(Y_mean: np.ndarray, 
        Y_var: np.ndarray, 
        beta: int = 2
        ) -> float:
    """Upper confidence bound acquisition score

    Parameters
    ----------
    Y_mean : np.ndarray (M,1)
        the mean predicted y values
    Y_var : np.ndarray (M,1)
        the variance of the mean predicted y values
    beta : int (Default = 2)
        the number of standard deviations to add to Y_mean

    Returns
    -------
    np.ndarray (M,1)
        the upper confidence bound acquisition scores
        
    ------
    
    https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc06805e
    The authors showed that UCB and greedy show the best performance for 
    virtual screening methods
    
    returns the upper confidence bound acquisition score calculated via the 
    means and variance for a list of samples
    """
    return Y_mean + beta*np.sqrt(Y_var)


#-----------------------------------------------------------------------------#  


def greedy(Y_mean: np.ndarray) -> np.ndarray:
    """Greedy acquisition score 

    Parameters
    ----------
    Y_mean : np.ndarray (M,1)
        the mean predicted y values

    Returns
    -------
    np.ndarray (M,1)
        the greedy acquisition scores
        
    -------
    
    
    https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc06805e
    The authors showed that UCB and greedy show the best performance for 
    virtual screening methods
    
    Greedy just returns the input value without any transformation as greedy 
    means using the label value directly
    """
    return Y_mean

