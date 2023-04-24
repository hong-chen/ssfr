from setuptools import setup, find_packages
import os

current_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(current_dir, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
     name = 'ARG-SSFR',
     version = '0.1',
     description = 'Python tools for Solar Spectrum Flux Radiometer (SSFR).',
     long_description = long_description,
     classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        ],
     keywords = 'SSFR utilities data calibration',
     url = 'https://gitlab.com/hong-chen/ssfr',
     author = 'Hong Chen',
     author_email = 'me@hongchen.cz',
     license = 'MIT',
     packages = find_packages(),
     install_requires = ['nose', 'numpy', 'scipy', 'h5py', 'matplotlib', 'pysolar'],
     python_requires = '~=3.7',
     include_package_data = True,
     zip_safe = False
     )
