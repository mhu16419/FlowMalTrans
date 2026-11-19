#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='flowmaltrans',
    version='1.0',
    description='FlowMalTrans: unsupervised cross-ISA binary code translation with a flow-adapter architecture '
                '(built on the XLM code base of Lample & Conneau)',
    url='https://github.com/mhu16419/FlowMalTrans',
    author='Minghao Hu et al. (based on code by Guillaume Lample and Alexis Conneau)',
    packages=find_packages(include=['xlm', 'xlm.*']),
    package_data={'xlm': ['evaluation/multi-bleu.perl']},
    python_requires='>=3.8',
    install_requires=['torch>=2.0', 'numpy>=1.21,<2', 'sacrebleu>=2.0'],
)
