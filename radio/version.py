"""
package version
"""

from __future__ import absolute_import, division, print_function
from os.path import join as pjoin

# root = Path(pkg.__path__[0]).parent.absolute()
# readme = pjoin(root, 'README.md')

# def read_long_description(readme):
#     text_type = type(u"")
#     with io.open(readme, mode="r", encoding="utf-8") as fd:
#         return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"),
#                       fd.read())

# Format expected by setup.py and docs/conf.py: string of form "X.Y.Z"
VERSION_MAJOR = '0'
VERSION_MAJOR = '1'
VERSION_MICRO = ''  # use '' for first of series, number for 1 and above
VERSION_EXTRA = 'dev'
# VERSION_EXTRA = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [VERSION_MAJOR, VERSION_MAJOR]
if VERSION_MICRO:
    _ver.append(VERSION_MICRO)
if VERSION_EXTRA:
    _ver.append(VERSION_EXTRA)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = [
    "Development Status :: 3 - Alpha", "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: BSD-3-Clause",
    "Operating System :: OS Independent", "Programming Language :: Python",
    "Programming Language :: Python :: 3.9", "Topic :: Scientific/Engineering"
]

NAME = "radio"
MAINTAINER = "Eduardo Diniz"
MAINTAINER_EMAIL = "eduardojdiniz@gmail.com"
DESCRIPTION = ("radIO: Radiology IO")
# LONG_DESCRIPTION = read_long_description(readme)
LONG_DESCRIPTION = ""
URL = "http://github.com/eduardojdiniz/radio"
DOWNLOAD_URL = ""
LICENSE = "BSD-3-Clause"
AUTHOR = "Eduardo Diniz"
AUTHOR_EMAIL = "eduardojdiniz@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = VERSION_MAJOR
MINOR = VERSION_MAJOR
MICRO = VERSION_MICRO
VERSION = __version__
PACKAGE_DATA = {'radio': [pjoin('data', '*')]}
REQUIRES: list = []  # use environment.yml for conda, requirements.txt for pip
PYTHON_REQUIRES = ">= 3.9"
