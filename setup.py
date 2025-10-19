from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sleep-analysis-pipeline",
    version="1.4.0",
    author="Tim Liner",
    author_email="psy66@ya.ru",
    description="Automated Polysomnography Data Analysis System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PSG",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.7",
    install_requires=[
        "mne>=1.0.0",
        "numpy>=1.21.0", 
        "scipy>=1.7.0",
    ],
)