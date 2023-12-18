#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:48:32 2023

@author: joshuahesse
"""
from setuptools import setup, find_packages
setup(name='DataValuationPlatform',
    version='0.2',
    packages=find_packages(exclude=["Experiments*", ".ipynb_checkpoints*", "DataValuationPlatform.egg-info*"]),
   include_package_data=True,
    package_data={
        '': ['Datasets/*'],
    },
   author='Joshua Hesse',
   author_email='your.email@example.com',
   description='A short description of your package',
   long_description=open('README.md').read(),
   long_description_content_type='text/markdown',
    url='https://github.com/JoshuaHesse/DataValuationPlatform')
