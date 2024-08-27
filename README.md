# GPU-Accelerated Levenshtein Distance Calculator

This repo holds a course project for Cyberinfrastructure at RIT. This project implements a GPU-accelerated version of the Levenshtein distance algorithm using PyCUDA. It's designed to efficiently compute the edit distance between two large strings, making it particularly useful for applications in bioinformatics, natural language processing, and data analysis. Please check the [project report](AcceleratingLevenshteinEditDistanceUsingGPUs.pdf) in this repo for more information.

## Background

### Levenshtein Distance

The Levenshtein distance, also known as edit distance, is a string metric for measuring the difference between two sequences. It represents the minimum number of single-character edits (insertions, deletions, or substitutions) required to change one word into another. This algorithm has various applications, including:

- Spell checking
- DNA sequence alignment
- Plagiarism detection
- Speech recognition

### PyCUDA

PyCUDA is a Python wrapper for NVIDIA's CUDA parallel computation API. It allows Python programmers to easily write GPU-accelerated algorithms by leveraging NVIDIA's CUDA architecture. PyCUDA provides:

- Access to NVIDIA's CUDA API from Python
- Automatic code generation and compilation
- Easy integration with NumPy

## Features

- GPU-accelerated Levenshtein distance calculation
- Ability to process large strings 
- Integration with MATLAB file formats for input data
- Performance measurement for execution time

## Requirements

- Python 3.x
- NVIDIA GPU with CUDA support
- PyCUDA
- NumPy
- SciPy

## Installation

1. Ensure you have an NVIDIA GPU with CUDA support.
2. Install the CUDA Toolkit and compatible GPU drivers.
3. Install the required Python packages:

```bash
pip install pycuda numpy scipy
```

## Usage

1. Place your input MATLAB (.mat) files in the appropriate directory.
2. Update the file paths in the script to point to your input files.
3. Run the script:

```bash
python levenGPU_demo.py
```

## How It Works

1. The script loads two strings from MATLAB files.
2. It allocates managed memory on the GPU for input strings and computation matrix.
3. A CUDA kernel is defined to perform the Levenshtein distance calculation.
4. The computation is split into multiple passes to handle large strings.
5. The final Levenshtein distance is extracted from the result matrix.

## Performance

This GPU-accelerated implementation can significantly outperform CPU-based versions, especially for large strings. The exact performance gain depends on the GPU hardware and the size of the input strings.

## Limitations

- Requires NVIDIA GPU with CUDA support.


## License

[MIT License](LICENSE)

## Acknowledgments

This project was inspired by the need for fast edit distance calculations in large-scale data processing tasks. Special thanks to the PyCUDA development team for providing the tools to make GPU acceleration accessible in Python.