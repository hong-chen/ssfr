from setuptools import setup, find_packages
import os

current_dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(current_dir, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
     name = 'SSFR-util',
     version = '0.1',
     description = 'Python tools for Solar Spectrum Flux Radiometer (SSFR).',
     long_description = long_description,
     long_description_content_type = 'text/x-rst',
     classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        ],
     keywords = 'SSFR utilities data calibration process',
     url = 'https://github.com/hong-chen/SSFR-util',
     author = 'Hong Chen, Sebastian Schmidt',
     author_email = 'me@hongchen.cz, sebastian.schmidt@lasp.colorado.edu',
     license = 'MIT',
     packages = ['ssfr_util'],
     install_requires = ['nose', 'numpy', 'scipy', 'h5py'],
     python_requires = '~=3.6',
     # scripts = ['bin/ssfr'],
     include_package_data = True,
     zip_safe = False
     )
