"""
do  python setup.py build develop
"""

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup
import torchvision
setup(
    name="net",
    version="1.0",
    author="JackY",
    url="unknown",
    description="Anomaly Detection in Video Sequence",
    install_requires=[
        "yacs>=0.1.6",
        "pyyaml>=5.1",
        "matplotlib",
        "termcolor>=1.1",
        "simplejson",
        "tqdm",
        "psutil",
        "matplotlib",
        "torchvision>=0.4.2",
        "sklearn"
    ],
    packages=find_packages(exclude=("configs")),
)



