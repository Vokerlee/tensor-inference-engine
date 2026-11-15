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

Initial project setup. Engine implementation and public APIs are planned.

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

## License

Licensed under the [Apache License 2.0](LICENSE).
