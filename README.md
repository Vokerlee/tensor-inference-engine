# Tensor Inference Engine

A modular tensor inference engine for hardware-aware research, numerical
algorithms, and experimental execution backends.

## Purpose

Provide reusable infrastructure for investigating how tensor computations are
represented, transformed, and executed under hardware constraints. The scope
includes inference operators and algorithms across model domains.

## Design goals

- **Independent abstractions:** lower-level components remain independent of
  model-specific logic, evaluation tasks, and research workflows.
- **Explicit execution constraints:** represent supported operation sizes,
  layouts, numerical formats, accumulation behavior, and buffering requirements.
- **Extensible algorithms:** support alternative computation methods and
  adaptive policies through reusable interfaces.
- **Defined numerical behavior:** make rounding, approximation, and partial-sum
  handling explicit so their effects can be evaluated in complete models.
- **Interchangeable backends:** separate algorithm semantics from reference
  implementations and optimized hardware-specific execution.

## Status

The native CUDA build and toolchain smoke check are available. Operator-specific
behavior and validation are documented alongside each implementation.

[Grouped representation references and CUDA preparation](docs/selection-reference.md)
cover activation formats,2:4 mask evaluation and block precision selection.

## Native CUDA build

Use Linux x86_64, Python 3.12, CMake 3.24 or newer, and a C++20 host compiler.
After installing the Python environment below, bootstrap the project-local
CUDA 12.8.1 compiler components and build:

```sh
python3.12 scripts/fetch_cuda_toolkit.py
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja"
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

The bootstrap verifies NVIDIA's redistributable SHA256 hashes and installs
under ignored `.toolchains/`; it does not install or change a system driver.
An existing toolkit can be selected with `-DCMAKE_CUDA_COMPILER=/path/to/nvcc`.
GPU access is required to run the CUDA tests; CTest reports a skip when no
device is available. A skipped test is not GPU validation.

Code generation defaults to SM89. Override `-DCMAKE_CUDA_ARCHITECTURES=...`
for another supported GPU. CUTLASS v4.8.0 headers are downloaded by CMake from
commit `098de2a652cf8f00fd70b2df54051c7eccbb855a`, with an archive SHA256 check.
The C++ core does not depend on Torch; its adapter is an optional build target.

## Python environment

The optional Torch integration and test environment targets CPython 3.12 on
Linux x86_64 with CUDA 12.8 runtime wheels. The generic C++ core remains
independent of Python and Torch.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

The requirements pin direct and transitive dependencies, including
`torch==2.11.0+cu128`, `pytest==8.4.2`, and `ninja==1.13.0`.
The official PyTorch CUDA wheel index is recorded in the file. Re-resolve and
commit the complete pinned set when changing a dependency.

The Python environment does not replace the native build toolchain. Compiling
CUDA extensions also requires a compatible Linux C++ compiler, CMake, and a
CUDA toolkit providing `nvcc`; the planned baseline uses CUDA 12.8. The NVIDIA
driver is managed separately. These environment files do not add an operator
implementation or claim that CUDA compilation has been validated.

## Formatting

The pinned requirements include clang-format 22.1.8 for C++/CUDA and gersemi
0.28.0 for CMake. Both use four spaces per indentation level. The scripts
prefer tools installed in this repository's `.venv` and format tracked files
only, excluding downloaded dependencies and build outputs.

```sh
scripts/format-cpp.sh
scripts/format-cpp.sh --check
scripts/format-cmake.sh
scripts/format-cmake.sh --check
scripts/install-hooks.sh
```

Use these scripts for formatting, including `.cu` and `.cuh` files. Style lives
in `.clang-format` and `.gersemirc`. The optional hook runs both format checks
before commits; install the pinned requirements before enabling it.

## License

Licensed under the [Apache License 2.0](LICENSE).
