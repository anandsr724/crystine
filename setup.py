from setuptools import setup , find_packages

classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]


setup(
    name='crystine',
    version='0.2.2',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'swap1=crystine.swap1:main',
            'swap2=crystine.swap2:main',
            'crystine-ginfo=crystine.ginfo:main',
            'crystine-gmass=crystine.gmass:main',
            
        ],
    },
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "matplotlib>=3.2.0",
        "pymatgen",
       
    ],
    author='Anand',
    author_email='anandsr724@gmail.com',
    description='A data analysis library for computational chemistry',
    python_requires='>=3.6',
)
