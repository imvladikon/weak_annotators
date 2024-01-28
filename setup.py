#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib
import os
from pathlib import Path

import setuptools

HERE = Path(__file__).parent

__package_name__ = 'weak_annotators'
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def import_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_version(package_name):
    version = import_file('version',
                          os.path.join(__location__, package_name, 'version.py'))
    return version.__version__


__version__ = get_version(__package_name__)


def read_requirements(reqs_path):
    with open(reqs_path, encoding='utf8') as f:
        reqs = [line.strip() for line in f
                if not line.strip().startswith('#') and not line.strip().startswith('--')]
    return reqs


if __name__ == '__main__':
    setuptools.setup(name=__package_name__,
                     version=__version__,
                     author='Vladimir Gurevich',
                     description='Weak annotators for information extraction (NER)',
                     long_description=(HERE / 'README.md').read_text(),
                     long_description_content_type='text/markdown',
                     url='https://github.com/imvladikon/weak_annotators',  # noqa
                     packages=setuptools.find_packages(exclude=(
                         'tests',
                         'tests.*',
                     )),
                     classifiers=[
                         'Programming Language :: Python :: 3',
                         'Topic :: Scientific/Engineering'
                     ],
                     python_requires='>=3.9',
                     package_dir={__package_name__: __package_name__},
                     package_data={
                         __package_name__: ['models/*/', 'models/*/*', 'assets/*']
                     },
                     include_package_data=True,
                     install_requires=read_requirements(HERE / 'requirements.txt'))
