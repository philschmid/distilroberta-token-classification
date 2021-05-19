from __future__ import absolute_import

from glob import glob
import os
from os.path import basename
from os.path import splitext

from setuptools import find_packages, setup

install_requires = [
    "sagemaker",
]

extras = {}

extras["tests"] = [
    "pytest",
]

extras["quality"] = [
    "black>=21.5b1",
    "isort>=5.5.4",
    "flake8>=3.8.3",
]


setup(
    name="distilroberta_token_classification",
    version="0.0.1",
    description="Training Distilroberta on token classification",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    extras_require=extras,
    python_requires=">=3.6.0",
    author="Philipp Schmid",
    author_email="philipp@huggingface.co",
    license="Apache License 2.0",
)
