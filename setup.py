#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os

from setuptools import setup, find_packages

def read(fname):
    file_path = os.path.join(os.path.dirname(__file__), fname)
    return codecs.open(file_path, encoding="utf-8").read()

setup(
    name='flimtools',
    version='0.0.1',
    author='Jay Unruh',
    description='A set of tools to do linear unmixing and phasor analysis on FLIM data.',
    url='https://github.com/jayunruh/FLIM_tools',
    license='GNU GPLv2',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=["numpy","numba","matplotlib"],
    py_modules=['linleastsquares','flimtools'],
)
