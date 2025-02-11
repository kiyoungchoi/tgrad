from setuptools import setup, find_packages

setup(
    name="tgrad",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch"
    ],
    author="Your Name",
    description="A tensor computation library",
    python_requires=">=3.6",
) 