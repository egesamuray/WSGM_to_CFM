# setup.py
from setuptools import setup, find_packages

setup(
    name="improved-diffusion",
    version="0.0.0",
    packages=find_packages(exclude=["tests", "notebooks"]),
    install_requires=[
        "torch",
        "tqdm",
        "blobfile>=1.0.5",
    ],
    extras_require={
        # Wavelet code needs PyWavelets; keep it optional so curvelet users can install cleanly.
        "wavelet": ["PyWavelets>=1.5.0"],
    },
    python_requires=">=3.9",
)
