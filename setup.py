import os
from setuptools import setup, find_packages, Extension

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "mlr",
    version = "0.0.1",
    author = "Kenneth Pierce",
    author_email = "ken.pierce+github@gmail.com",
    description = ("Machine Learning Research"),
    keywords = "",
    url = "https://github.com/KennethPierce/MLResearch/",
    packages=find_packages(),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: Alpha",
    ],
)