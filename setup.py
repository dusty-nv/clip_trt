#!/usr/bin/env python3
from setuptools import setup, find_packages
from clip_trt.version import __version__

setup(
    name="clip_trt",
    version=__version__,
    packages=find_packages()
)
