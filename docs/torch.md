# Optional Torch operator

Build after installing the root requirements and native toolchain:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_MAKE_PROGRAM="$PWD/.venv/bin/ninja" \
  -DTIE_BUILD_TORCH=ON -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build --parallel 2
.venv/bin/python -m pytest tests/test_torch.py
```

Load the operator once; input tensors stay on their existing CUDA device:

```python
import torch

torch.ops.load_library("build/libtensor_inference_engine_torch.so")
a = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
b = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
c = torch.ops.tensor_inference_engine.gemm(a, b)  # BF16 output
c_fp32 = torch.ops.tensor_inference_engine.gemm(a, b, True)
```

The adapter allocates the output through Torch, passes borrowed input/output
views into the independent CUDA core, and uses Torch's current CUDA stream.
Inputs must be BF16, 2D, on the same CUDA device, and either row/column-major
with non-overlapping leading dimensions. There is no implicit packing/copy.
Use inference mode for tensors requiring gradients; backward is unsupported.

Meta registration supports shape inspection and compilation checks. Tests
cover `torch.library.opcheck`, graph replay, nondefault streams, transposes,
pitched/offset storage, empty reductions and a linear-plus-activation path.
BF16 results are compared with declared tolerances; no bitwise or target-ASIC
arithmetic equivalence is claimed. Set `TIE_TORCH_LIBRARY` to test another build.
