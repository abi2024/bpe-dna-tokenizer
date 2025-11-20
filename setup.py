from setuptools import setup, find_packages

setup(
    name="bpe-dna-tokenizer",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Byte Pair Encoding tokenizer for DNA sequences",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/abi2024/bpe-dna-tokenizer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
    ],
)