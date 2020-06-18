import io
import os
from pathlib import Path
from setuptools import find_packages, setup


# package meta-data
NAME = "regression_pipeline"
DESCRIPTION = "Train and deploy regression model"
URL = "https://github.com/RudyWilliams/deploy-ml-udemy-coursework"
EMAIL = "rudysw05@knights.ucf.edu"
AUTHOR = "Rudy Williams"
REQUIRES_PYTHON = ">=3.8.0"


def list_reqs(fname="requirements.txt"):
    with open(fname) as fd:
        return fd.read().splitlines()


here = os.path.abspath(os.path.dirname(__file__))

# import the readme to use as the long-description
# note: this will only work if README.md is present in your MANIFEST.in file
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# load the package's __version__.py module as a dictionary
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / NAME
about = dict()
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"regression_model": ["VERSION"]},
    install_requires=list_reqs(),
    extra_require={},
    include_package_data=True,
    license="MIT",
    classifiers=[
        # Trove classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
