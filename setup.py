from setuptools import setup
import sys

if sys.version_info.major != 2:
    print("This package is only compatible with Python 2.")
    exit(1)

extras = {
    'plots': ['plotly']
}

setup(
    name='pedestrian_prediction',
    version='0.1',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/sirspinach/pedestrian_prediction',
    author="Steven H. Wang",
    author_email='wang.steven.h@gmail.com',
    packages=['pp'],
    install_requires=[
        'enum34',
        'scipy',
        'scikit-learn',
        ],
    extras_require=extras,
)
