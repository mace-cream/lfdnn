from setuptools import setup
import setuptools
from lfdnn import __version__
with open("README.md") as fh:
    long_description = fh.read()
    
if __name__ == '__main__':
    setup(name = 'lfdnn',
          version = __version__,
          description = 'Learning from data Neural Network',
          author = 'zhaofeng-shu33',
          author_email = '616545598@qq.com',
          url = 'https://github.com/mace-cream/lfdnn',
          maintainer = 'zhaofeng-shu33',
          maintainer_email = '616545598@qq.com',
          long_description = long_description,
          long_description_content_type="text/markdown",          
          install_requires = ['numpy'],
          license = 'Apache License Version 2.0',
          packages=setuptools.find_packages(),
          classifiers = (
              "Development Status :: 4 - Beta",
              "Programming Language :: Python :: 3.7",
              "Programming Language :: Python :: 3.6",
              "Operating System :: OS Independent",
          ),
          )