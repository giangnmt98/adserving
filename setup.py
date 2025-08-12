#!/usr/bin/env python3
"""Setup script for adserving package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read file requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# Read the contents of the README file
with open(Path(BASE_DIR, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Development dependencies
dev_requires = [
    "pytest>=7.4.0",
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
    install_requires=required_packages,
    extras_require={
        "dev": dev_requires + required_packages,
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
