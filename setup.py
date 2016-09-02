#!/usr/bin/env python
from setuptools import setup
from distutils.command.build import build
import subprocess

class build_feff(build):
    def run(self):
        subprocess.call('make -C feff', shell=True)

class build_custom(build):
    def run(self):
        self.run_command('build_feff')
        build.run(self)

setup(
    name='expectra',
    version='0.1.0',
    author='Samuel T. Chill',
    author_email='samchill@gmail.com',
    url='https://github.com/SamChill/expectra',
    packages=['expectra'],
    scripts=[
        'bin/expectra',
        'bin/xafsft',
        'bin/harmonic-sampler',
        'bin/pdfstats',
        'feff/feff',
    ],
    license='LICENSE.txt',
    description='Code for EXAFS calculations from dynamics or normal modes',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy >= 1.5.0',
        'mpi4py >= 1.3',
    ],
    cmdclass={
        'build'      : build_custom,
        'build_feff' : build_feff,
    },
)
