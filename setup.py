from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="emns",
    version="1.0.0",
    author="Cahit Karahan",
    author_email="cksoftwaresystems@gmail.com",
    description="Evolvable Modular Neural Systems - A Self-Organizing, Resistance-Governed Neural Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["lib_and_demo"],
    install_requires=requirements,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords="neural networks, evolutionary computation, self-organization, neuromorphic computing",
) 