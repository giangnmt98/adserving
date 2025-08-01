#!/usr/bin/env python3
"""Setup script for adserving package."""

import os

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Core dependencies (production)
install_requires = [
    "mlflow==3.1.4",
    "ray[serve]==2.48.0",
    "pandas>=1.5.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "psutil>=5.9.0",
    "GPUtil>=1.4.0",
    "PyYAML>=6.0",
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "prometheus-client>=0.19.0",
    "structlog>=23.2.0",
]

# Development dependencies
dev_requires = [
    "black==24.1.1",
    "flake8==7.0.0",
    "mypy==1.8.0",
    "isort==5.13.2",
    "types-PyYAML==6.0.12.12",
    "pre-commit==3.6.0",
    "pylint==3.3.7",
    "types-requests",
]

setup(
    name="adserving",
    version="0.1.0",
    author="Data Science Team",
    author_email="",
    description="A machine learning model serving system with advanced routing and monitoring capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.11",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
    },
    include_package_data=True,
    package_data={
        "adserving": ["*.yaml", "*.yml", "*.json"],
    },
    entry_points={
        "console_scripts": [
            # Add console scripts here if needed
        ],
    },
    zip_safe=False,
)
