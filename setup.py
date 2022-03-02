from setuptools import setup, find_packages
import glob, os


print([f for f in glob.glob('data/**/*', recursive=True) if os.path.isfile(f) ])

setup(
    name='LiSA',
    version='1.0.0',    
    description='A lightweight HI source finding for next generation radiosurveys',
    url='https://github.com/epfl-radio-astro/LiSA',
    author='Emma Tolley, Aymeric Galan, Damien Korber, Mark Sargent', # Can be reordered as you wish
    author_email='ADD EMAIL ADRESSES',
    license='GPL-3.0',
    packages=['lisa', 'lisa.ai', 'lisa.nht', 'lisa.utils', 'lisa.pipelines'],
     package_dir={
        'lisa': 'modules',
        'lisa.ai': 'modules/ai',
        'lisa.nht': 'modules/nht',
        'lisa.utils': 'modules/utils',
        'lisa.pipelines': 'pipelines'
        },
    #packages=find_packages(), #['lisa', 'modules', 'pipelines'],#, 'modules', 'modules.ai', 'modules.nht', 'modules.util', ],
    include_package_data=True,
    install_requires=[
        'astropy==5.0.1',
        'h5py==3.6.0',
        'iminuit==2.0.0',
        'keras==2.8.0',
        'numpy',
        'sklearn',
        'tensorflow==2.8.0',
        'numba',
        'python-pysap',
        'lmfit',
        'scipy',
        'matplotlib'
    ],
    data_files=[f for f in glob.glob('data/**/*', recursive=True) if os.path.isfile(f)],


    classifiers=[ # CHANGE THIS
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
)