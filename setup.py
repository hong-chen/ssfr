from setuptools import setup, find_packages
import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(current_dir, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
     name = 'ssfr',
     version = '0.0.1',
     description = 'SSFR Software Package',
     long_description = long_description,
     classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        ],
     keywords = 'SSFR',
     url = 'https://github.com/hong-chen/ssfr',
     author = 'Hong Chen',
     author_email = 'hong.chen@lasp.colorado.edu, sebastian.schmidt@lasp.colorado.edu, yu-wen.chen@colorado.edu, ken.hirata@colorado.edu',
     license = 'MIT',
     packages = find_packages(),
     install_requires = [
         'numpy',
         'scipy',
         'h5py',
         'matplotlib',
         'pysolar',
         ],
     python_requires = '~=3.9',
     scripts = ['bin/sks2mp4', 'bin/fits2jpg'],
     include_package_data = True,
     zip_safe = False
     )
