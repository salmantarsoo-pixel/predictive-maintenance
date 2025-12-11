from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="predictive-maintenance-ml",
    version="1.0.0",
    author="Salman Tarsoo",
    author_email="salman.tarsoo@gmail.com",
    description="Production-ready machine learning system for predictive maintenance using the AI4I 2020 dataset",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/salmantarsoo-pixel/predictive-maintenance",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0.0",
    ],
)
