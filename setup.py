#!/usr/bin/env python
"""
package setup.
"""

import sys
from pathlib import Path
from setuptools import find_packages, setup

PACKAGES = find_packages(exclude=("tests",), include=["radio*"])

# Get version and release info
ver_file = Path("radio", "version.py")
with ver_file.open() as fid:
    exec(fid.read())

# Give setuptools a hint to complain if it's too old a version
# 24.2.0 added the python_requires option
# Should match pyproject.toml
SETUP_REQUIRES = ["setuptools >= 24.2.0"]
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ["wheel"] if "bdist_wheel" in sys.argv else []

ENTRY_POINTS = {"console_scripts": ["radio=radio.app:app"]}

opts = dict(
    name=NAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    download_url=DOWNLOAD_URL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    platforms=PLATFORMS,
    version=VERSION,
    packages=PACKAGES,
    package_data=PACKAGE_DATA,
    entry_points=ENTRY_POINTS,
    install_requires=REQUIRES,
    python_requires=PYTHON_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    requires=REQUIRES,
)

if __name__ == "__main__":
    setup(**opts)
