from setuptools import setup, find_packages

setup(
    name="Rank-lq-CMA-ES",
    version="0.1.0",
    description="This module implements an invariant surrogate assisted CMA-ES under strictly increasing transformations of the objective function, using a linear quadratic model as surrogate on rank values instead of fitness values. (see paper) https://doi.org/10.1145/3712255.3726606",
    author="Mohamed GHARAFI",
    author_email="mohamed@mgharafi.me",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "scipy",
        "cma",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
