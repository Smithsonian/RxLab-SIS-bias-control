"""Setup script for RxLab-SIS-bias

Usage (from root folder):

    python3 -m pip install -e .
    sudo python3 -m pip install -e .

"""

import io
from os import path

from setuptools import find_packages, setup

import sisbias


def read(*filenames):
    buf = []
    root = path.abspath(path.dirname(__file__))
    for filename in filenames:
        with io.open(path.join(root, filename), encoding='utf-8') as f:
            buf.append(f.read())
    return '\n'.join(buf)


setup(
    name="RxLab-SIS-bias",
    version=sisbias.__version__,
    author="John Garrett",
    author_email="john.garrett@cfa.harvard.edu",
    description="Control the SIS bias via the MCC DAQ device",
    url="https://github.com/Smithsonian/RxLab-SIS-bias/",
    packages=find_packages(),
    install_requires=[
        'appdirs', 'numpy', 'matplotlib'
    ],
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    scripts=[
        'bin/sisbias',
        'bin/sisbias-init-config-v0',
        'bin/sisbias-init-config-v3',
        'bin/sisbias-init-param',
    ],
)
