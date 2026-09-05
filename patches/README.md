# Patches: Windows single-GPU CUDA build (no NCCL)

This directory documents and reproduces the patch set that makes LightGBM's
CUDA build work on **Windows** for a **single GPU**, without NCCL (which has
no official Windows distribution). See the repository README for the summary.

Full documentation (Chinese): **[README.zh.md](README.zh.md)**

## What is patched

| File | Change | Why |
|---|---|---|
| `CMakeLists.txt` | `find_package(NCCL REQUIRED)` → `QUIET`, falls back to the bundled stub library exposed as `NCCL::NCCL`; adds `-DFMT_UNICODE=0` to CUDA flags | NCCL is hard-required since #6138 but unavailable on Windows; every NCCL call site only runs with `num_gpu > 1`, so a stub is safe (it aborts loudly if ever invoked) |
| `cmake/nccl-stub/` | new — minimal `nccl.h` subset + aborting implementations | satisfies the compiler and linker; `ncclUniqueId` keeps its 128-byte layout |
| `src/boosting/cuda/nccl_gbdt.hpp` | `#include <pthread.h>` guarded with `!_WIN32` (unused include) | pthread does not exist on Windows |
| `include/LightGBM/meta.h` | `const` → `constexpr` for device-code constants | nvcc 13.x requires inlinable constants in device code |
| `src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp` | explicit `static_cast<size_t>` in `std::max` | Win64 `size_t` vs 32-bit `unsigned long` breaks deduction on newer MSVC |
| `external_libs/fmt` submodule | `U""` char32_t literal → integer array | nvcc's EDG front end rejects large char32_t literals |

## Reapplying after an upstream update

Baselines are upstream **stable release tags** (not main snapshots); fork tags
are named `<baseline>-cuda<cuda-version>[-rN]`. Steps, conflict points and the
build commands are described in [README.zh.md](README.zh.md).
