from setuptools import setup

setup(
    name='expectra',
    version='0.0.1',
    author='Samuel T. Chill',
    author_email='samchill@gmail.com',
    packages=['expectra'],
    scripts=['bin/expectra', 'bin/xafsft'],
    url='',
    license='LICENSE.txt',
    description='Code for EXAFS calculations from dynamics or normal modes',
    long_description=open('README.txt').read(),
    install_requires=[
        'numpy >= 1.5.0',
        'mpi4py >= 1.3',
    ],
)
