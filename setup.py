#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:48:32 2023

@author: joshuahesse
"""
from setuptools import setup, find_packages
setup(name='DataValuationPlatform',
    version='1.1',
    packages=['DataValuationPlatform','DataValuationPlatform.models',
                                'DataValuationPlatform.models.catboost',
                                'DataValuationPlatform.models.mvsa',
                                'DataValuationPlatform.models.tracin',
                                'DataValuationPlatform.models.utils',
                            ],
   author='Joshua Hesse')
