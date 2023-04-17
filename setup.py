from setuptools import setup
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='aalpy',
    version='1.3.1',
    packages=['aalpy', 'aalpy.base', 'aalpy.SULs', 'aalpy.utils', 'aalpy.oracles', 'aalpy.automata',
              'aalpy.learning_algs', 'aalpy.learning_algs.stochastic', 'aalpy.learning_algs.deterministic',
              'aalpy.learning_algs.non_deterministic',
              'aalpy.learning_algs.stochastic_passive', 'aalpy.learning_algs.deterministic_passive'],
    url='https://github.com/DES-Lab/AALpy',
    license='MIT',
    license_files=('LICENSE.txt',),
    long_description = long_description,
    long_description_content_type='text/markdown',
    author='Edi Muskardin',
    author_email='edi.muskardin@silicon-austria.com',
    description='An active automata learning library',
    classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent"
    ],
    install_requires=['pydot'],
    python_requires=">=3.6",
)

# python setup.py sdist
# pip wheel . -w dist
# twine upload dist/*

# for test pypi
# twine upload --repository testpypi dist/*