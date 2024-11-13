from setuptools import setup, find_packages

setup(
    name='data_store',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'sentence-transformers',
        'langchain',
        'langchain-community',
        'langchain-huggingface',
        'langchain_chroma',
        'pymupdf',
    ],
)
