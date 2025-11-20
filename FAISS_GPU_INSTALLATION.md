# FAISS GPU Installation Guide for CUDA 13

## Issue

`faiss-gpu` is **not available on PyPI** via pip. The package uses `faiss-cpu` by default, which works reliably but only provides CPU acceleration.

## Solution: Install FAISS GPU via Conda

The recommended way to install FAISS GPU (especially for CUDA 13) is using Conda:

### Option 1: Install FAISS GPU via Conda (Recommended)

```bash
# Install faiss-gpu via conda
conda install -c pytorch faiss-gpu

# Then install OpenClio (it will use the conda-installed faiss-gpu)
pip install -e .
```

### Option 2: Mixed Environment (Conda + pip)

If you're using a pip virtual environment but want FAISS GPU:

```bash
# Create conda environment with FAISS GPU
conda create -n openclio python=3.10
conda activate openclio
conda install -c pytorch faiss-gpu

# Install other dependencies via pip
pip install -e .
```

### Option 3: Build from Source

If conda is not an option, you can build FAISS from source:

```bash
# Prerequisites: CUDA 13 toolkit, CMake, etc.
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build -DFAISS_ENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"
make -C build -j faiss
pip install build/faiss/python/
```

## Verification

After installation, verify GPU support:

```python
import faiss

# Check if GPU is available
print(f"Number of GPUs: {faiss.get_num_gpus()}")

# Test GPU index creation
if faiss.get_num_gpus() > 0:
    d = 64
    res = faiss.StandardGpuResources()
    index_cpu = faiss.IndexFlatL2(d)
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
    print("GPU support is working!")
else:
    print("No GPU detected - using CPU")
```

## Current Configuration

OpenClio is configured to use `faiss-cpu` by default, which:
- ✅ Works reliably via pip
- ✅ Provides CPU acceleration
- ⚠️ Does not use GPU acceleration

To enable GPU acceleration:
1. Install `faiss-gpu` via conda (see above)
2. The existing code will automatically detect and use GPU if available
3. No code changes needed - FAISS automatically uses GPU when `faiss-gpu` is installed

## Notes

- FAISS will automatically use GPU if `faiss-gpu` is installed, even if `faiss-cpu` is also installed
- The `faissKMeans.py` code doesn't need changes - it will use GPU automatically
- For CUDA 13, ensure you have compatible CUDA toolkit and drivers installed
- Python 3.10-3.11 generally have better FAISS GPU support than Python 3.12

