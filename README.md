![Intel Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Intel_logo_%282006-2020%29.svg/200px-Intel_logo_%282006-2020%29.svg.png)

# Hands-on SIMD Programming with C++

> An example-driven guide to SIMD programming techniques.

## Overview

This repository demonstrates practical SIMD (Single Instruction, Multiple Data) programming in C++ through progressive examples. It focuses on performance-critical techniques that leverage parallel data processing capabilities of modern CPUs.

| Feature | Description |
|---------|-------------|
| **Target Audience** | Beginner to intermediate C++ programmers |
| **Instruction Sets** | SSE, AVX, AVX2, AVX-512 |
| **Performance Gains** | Up to 23x speedup in example benchmarks |

## Repository Structure

```
├── 01_Basics/               # Fundamental SIMD concepts
│   ├── 01_importing_simd/   # Headers and instruction sets
│   ├── 02_initializing_data/# Working with SIMD data types
│   ├── 03_binding_with_unions/ # Data access techniques
│   └── 04_loading_data/     # Load/store operations
├── 02_Computations/         # Mathematical operations
│   ├── 01_simple_maths/     # Basic arithmetic operations
│   └── 02_dot_product/      # Vector dot products
├── 03_Examples/             # Real-world applications
│   ├── 01_conditional_code/ # Branching with SIMD masks
│   ├── 02_quadratic_equations/ # Parallel equation solving
│   ├── 03_data_types/       # Type conversions and operations
│   └── 04_image_processing/ # Image manipulation algorithms
└── include/                 # Utility headers
    └── simd_utils.h         # Common helper functions
```

## Key Features

- **Comprehensive Coverage**: From basic concepts to advanced techniques
- **Performance Benchmarks**: Each example includes scalar vs. SIMD comparisons
- **Practical Applications**: Real-world examples demonstrating SIMD benefits
- **Progressive Learning**: Step-by-step approach with increasing complexity

## Quick Start

### Prerequisites
- Modern C++ compiler with AVX2 support (GCC 4.9+, Clang 3.6+, MSVC 2015+)
- Basic C++ knowledge

### Build & Run

```bash
# Clone repository
git clone https://github.com/yuninxia/hands-on-simd-programming.git
cd hands-on-simd-programming

# Build and run an example
cd 01_Basics/01_importing_simd
make
./simd_program

# View assembly output
make asm
cat main.s
```

## Core SIMD Techniques Covered

### Data Types & Initialization
- `__m256`, `__m256d`, `__m256i` vectors
- Zero, broadcast, and element-wise initialization
- Memory alignment considerations

### Mathematical Operations
- Vector arithmetic: `_mm256_add_ps()`, `_mm256_mul_ps()`, etc.
- Horizontal operations: `_mm256_hadd_ps()`
- Fused multiply-add: `_mm256_fmadd_ps()`

### Advanced Techniques
- Conditional processing with masks
- Type conversions between SIMD registers
- Non-temporal memory operations
- Parallel algorithm implementation

## Performance Highlights

| Example | Scalar | SIMD | Speedup |
|---------|--------|------|---------|
| Image Brightness | 1,225,747 μs | 52,418 μs | 23.38x |
| Square Root | 36,612 μs | 2,179 μs | 16.80x |
| Vector Addition | 17,010 μs | 2,786 μs | 6.11x |
| Quadratic Equations | 67,972 μs | 27,667 μs | 2.46x |

## Resources

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html) - Official reference for SIMD intrinsics
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/) - Detailed processor optimization guides
- [SIMD at Insomniac Games](https://deplinenoise.files.wordpress.com/2015/03/gdc2015_afredriksson_simd.pdf) - Game industry SIMD techniques

## License

MIT