from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dirac_graph",
    version="1.3.4",
    author="Mark Hale",
    license="MIT",
    description="Computational spectral geometry on finite graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pulquero/dirac_graph",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
