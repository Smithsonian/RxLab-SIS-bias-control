"""Setup script for RxLab-SIS-bias-control

Usage (from root folder):

    python3 -m pip install -e .

"""

import io
from os import path

from setuptools import find_packages, setup


def read(*filenames):
    buf = []
    root = path.abspath(path.dirname(__file__))
    for filename in filenames:
        with io.open(path.join(root, filename), encoding='utf-8') as f:
            buf.append(f.read())
    return '\n'.join(buf)


setup(
    name="RxLab-SIS-bias-control",
    version="0.0.1.dev",
    author="John Garrett",
    author_email="john.garrett@cfa.harvard.edu",
    description="Control the SIS bias board via an MCC DAQ device",
    url="https://github.com/Smithsonian/RxLab-SIS-bias/",
    packages=find_packages(),
    install_requires=[
        'appdirs', 'numpy', 'matplotlib', 'uldaq', 'scipy',
    ],
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
    ],
    scripts=[
        'bin/2sb',
        'bin/sisbias',
        'bin/sisbias-init-cal',
        'bin/sisbias-init-config-v0',
        'bin/sisbias-init-config-v3',
        'bin/sisbias-init-param',
    ],
)
