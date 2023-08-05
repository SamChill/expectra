#!/usr/bin/env python
from setuptools import setup
from distutils.command.build import build
import subprocess, os, stat, shutil


class build_feff(build):
    def run(self):
        subprocess.call('make -C feff', shell=True)
        # ensure the binary is executable
        st = os.stat('feff/feff')
        os.chmod('feff/feff', st.st_mode | stat.S_IEXEC)
        # move to package dir
        shutil.move('feff/feff', 'expectra/feff')

class build_custom(build):
    def run(self):
        self.run_command('build_feff')
        build.run(self)

try:
   import pypandoc
   long_description = pypandoc.convert_file('README.md', 'rst')
except (IOError, ImportError):
   long_description = ''

description = 'Code for simulating EXAFS calculations from molecular ' \
              'dynamics trajectories or normal modes using FEFF'

setup(
    name='expectra',
    version='1.0.5',
    author='Samuel T. Chill',
    author_email='samchill@gmail.com',
    url='https://github.com/SamChill/expectra',
    packages=['expectra'],
    scripts=[
        'bin/expectra',
        'bin/xafsft',
        'bin/harmonic-sampler',
        'bin/pdfstats',
        'bin/feff',
    ],
    package_data={
        'expectra': ['feff'],
    },
    license='LICENSE.txt',
    description=description,
    long_description=long_description,
    install_requires=[
        'numpy >= 1.5.0',
        'mpi4py >= 1.3',
        'ase >= 3.11.0',
    ],
    cmdclass={
        'build'      : build_custom,
        'build_feff' : build_feff,
    },
)
