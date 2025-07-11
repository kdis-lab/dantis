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
    #  $ pip install dantis
    name='dantis',

    # Version
    version='0.0.1',

    # Description
    description='DANTIS brings together **50+ anomaly detection algorithms**—from classical statistical methods and machine learning to deep learning—under a **unified scikit-learn–style API**. It consolidates models from diverse libraries and includes proprietary algorithms developed by our team.',

    # Long description (README)
    long_description=long_description,

    # URL
    url='https://github.com/kdis-lab/dantis',

    # Author
    author='DaSCI, KDIS Lab, SCI2S',

    # Author email
    author_email='',

    # Keywords
    keywords=['Anomaly Detection', 'Time Series', 'Machine Learning', 'Deep Learning', 'Scikit-learn', 'Data Science'],

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