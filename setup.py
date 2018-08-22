import setuptools
import os

with open(os.path.join('convtt', "README.md"), "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="convtt",
    version="0.0.1",
    author="Bin Wang",
    author_email="wwwbbb8510@gmail.com",
    description="Convolutional Neural Netoworks Training Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wwwbbb8510/convtt.git",
    packages=setuptools.find_packages(),
    scripts=[
        'convtt/bin/convtt_train_densenet.py',
        'convtt/bin/convtt_train_dynamicnet.py',
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)