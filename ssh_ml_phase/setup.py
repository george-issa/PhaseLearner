from setuptools import setup, find_packages

setup(
    name="ssh_ml_phase",
    version="0.1.0",
    description="ML methods for detecting phase transitions in the optical SSH model",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "matplotlib",
    ],
    extras_require={
        "umap": ["umap-learn"],
    },
)
