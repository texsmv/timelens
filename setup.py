from setuptools import setup, find_packages

setup(
    name='timelens',  # Your package name
    version='0.1.0',  # Your package version
    packages=find_packages(), # Automatically find your packages
    entry_points={
        'console_scripts': [
            'timelens=timelens_cli:main', # This line is the important part
        ],
    },
    install_requires=[ # List your dependencies here
        'flask',
        'flask-cors',
        'numpy',
        'argparse' # Make sure argparse is included if you are using it in timelens_cli.py
    ],
)