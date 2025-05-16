from setuptools import setup, find_packages

setup(
    name="hapkert",
    version="1.0.0",
    author="Ryleigh Davis",
    author_email="",
    description="A Python package for Hapke radiative transfer calculations",
    long_description="HapkeRT - A Python package for Hapke radiative transfer calculations",
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
    ],
    package_data={
        "hapkert": ["mastrapa_data.txt"],
    },
)