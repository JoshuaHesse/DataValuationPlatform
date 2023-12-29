from setuptools import setup, find_packages

setup(
    name='DataValuationPlatform',
    version='1.3',
    packages=find_packages(exclude=["Experiments*", ".ipynb_checkpoints*", "DataValuationPlatform.egg-info*"]),
    include_package_data=True,
    install_requires=[
    'chembl-structure-pipeline==1.2.0',
    'gpflow==2.8.0',
    'imbalanced-learn==0.10.1',
    'jupyter',
    'ipykernel',
    'matplotlib==3.6.0',
    'pandas==1.4.0',
    'rdkit==2022.9.5',
    'scikit-learn==1.2.2',
    'scipy==1.8.1',
    'tensorflow==2.4.0',
    'tensorflow-probability==0.12.1',
    'tqdm==4.60.0',
    'lightgbm==3.3.5',
    'catboost==1.1.1',
    ],
    python_requires='==3.8.*',
    author='Joshua Hesse',
    author_email='joshua.hesse@tum.de',
    description='A modular platform to test data valuation methods on High Throughput Screen data applications',
    long_description=open('README_PyPI.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/JoshuaHesse/DataValuationPlatform',
    # Other parameters as needed
)

