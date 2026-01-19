"""Setup configuration for COSMO"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cosmo-finance",
    version="0.1.0",
    author="BK-Korea",
    description="AI-powered Yahoo Finance assistant for stock market analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "yfinance>=0.2.36",
        "langchain>=0.3.0",
        "langchain-openai>=0.2.0",
        "langchain-community>=0.3.0",
        "langgraph>=0.2.0",
        "chromadb>=0.5.0",
        "pydantic>=2.6.0",
        "pydantic-settings>=2.2.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "typer>=0.12.0",
        "rich>=13.7.0",
        "requests>=2.31.0",
        "tenacity>=8.2.3",
        "openai>=1.12.0",
    ],
    entry_points={
        "console_scripts": [
            "cosmo=src.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
