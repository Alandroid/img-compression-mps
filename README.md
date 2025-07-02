# img-compression-mps

A Python library for compressing and benchmarking N-dimensional scientific data (e.g. MRI, fMRI, video tensors) using **Matrix Product States (MPS)** and tensor network techniques.

---

## What It Does

This project allows you to:

- Encode high-dimensional arrays into an MPS structure.
- Apply compression via bond truncation (using SVD).
- Benchmark the performance via metrics like:
  - SSIM (2D–4D aware)
  - PSNR
  - Fidelity (state overlap)
  - Compression ratios (raw and gzipped)
- Use `DCT` or standard encoding modes.

---

## Project Structure

``` bash
src/
├── imgcompressionmps/
│   ├── ndmps.py               # Main NDMPS class (encode/compress)
│   └── utils/
│       ├── core.py            # Tensor encoding utilities
│       ├── filetools.py       # File loading, scaling, slicing
│       └── metrics.py         # SSIM, PSNR, overlap
├── evaluation/
│   ├── benchmark.py           # Benchmark orchestration
│   ├── run_benchmark.py       # Examples for MRI/fMRI datasets
│   └── results/               # Output folder for JSON results
tests/
    └── test_ndmps.py          # Unit tests for MPS operations
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Run a benchmark (example)
python src/imgcompressionmps/evaluation/main.py
```

---

## Example Output

The benchmarking pipeline saves results as JSON, including SSIM vs. compression ratio, which can be plotted with `paper_eval.ipynb`.

---

## Testing

Tests are located in `/tests`. Run them using:

```bash
pytest tests/
```

---

## Installation

```bash
pip install -e .
```

---

## Use Cases

- Low-rank compression of scientific data
- Comparing structured (MPS) vs. unstructured compression
- Physics-informed ML representations
- Efficient storage and visualization

---

## License

MIT License (open source)
