import setuptools

setuptools.setup(
    name = "openclio",
    version = "0.0.1",
    author = "Phylliida",
    author_email = "phylliidadev@gmail.com",
    description = "Open source version of Anthropic's Clio: A system for privacy-preserving insights into real-world AI use",
    url = "https://github.com/Phylliida/OpenClio.git",
    project_urls = {
        "Bug Tracker": "https://github.com/Phylliida/OpenClio/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = setuptools.find_packages(),
    python_requires = ">=3.6",
    install_requires = ["concave_hull", "tqdm", "umap-learn", "numba>=0.60.0", "cryptography", "dataclasses", "vllm>=0.6.0", "pandas", "sentence_transformers", "numpy", "scikit-learn", "scipy", "torch>=2.5.0", "cloudpickle", "setuptools", "pyarrow", "faiss-gpu"]
)
