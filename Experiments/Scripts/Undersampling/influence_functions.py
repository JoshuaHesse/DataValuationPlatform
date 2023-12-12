#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 12:59:13 2023

@author: joshua

This file contains the dictionary of all influence functions.
"""

from active_learning_mvsa import influences_mvsa
from active_learning_catboost import influences_catboost_self, influences_catboost_test
from active_learning_dvrl import influences_dvrl
from active_learning_tracin import influences_tracin, influences_tracin_pos, influences_tracin_self
from active_learning_shapley import influences_shapley 



#this dictionary allows influence prediction via indexing using the name of the
#influence function
influence_functions_dict = {"mvsa": influences_mvsa, "shapley": influences_shapley, "catboost_self": influences_catboost_self,
                            "catboost_test": influences_catboost_test, "tracin": influences_tracin,
                            "tracin_pos": influences_tracin_pos, "tracin_self": influences_tracin_self, "dvrl": influences_dvrl}