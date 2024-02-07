from setuptools import setup, find_packages

from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    "numpy>=1.20",
    "pyod>=1.1",
    "xgboost>=2.0",
    "combo>=0.1",
    "suod>=0.1.1",
    "tensorflow>=2.13",
    "torch>=2.1.2",
    "stumpy>=1.12.0",
    "statsmodels>=0.13.5"
]

setup(
    #  Project name.
    #  $ pip install anomaly_lib
    name='anomaly_lib',

    # Version
    version='1.0',

    # Description
    description='Library for ...',

    # Long description (README)
    long_description=long_description,

    # URL
    url='https://github.com/kdis-lab/anomaly_lib',

    # Author
    author='EN UNA SOLA LINEA',

    # Author email
    author_email='correos',

    # Keywords
    keywords=['Anomaly Detection'],

    # Packages
    package_dir={"": "./"}, 
    packages=find_packages(where="./", exclude=['app', 'docs', 'tests', 'examples', 'src', "setup.py"]),
    include_package_data=True,
    package_data={},
    # Test suite

    # Requeriments
    install_requires=install_requires,
    long_description_content_type='text/markdown'
)