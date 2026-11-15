#!/usr/bin/env python3
"""Install the pinned Linux x86_64 CUDA compiler components locally."""

import argparse
import hashlib
import json
import platform
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist/"
# NVIDIA redistrib_12.8.1.json. Only compiler, runtime and C++ headers are needed.
COMPONENTS = {
    "cuda_nvcc": (
        "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz",
        "9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3",
    ),
    "cuda_cudart": (
        "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz",
        "8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68",
    ),
    "cuda_cccl": (
        "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz",
        "0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b",
    ),
}


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def install(prefix):
    expected = {name: digest for name, (_, digest) in COMPONENTS.items()}
    marker = prefix / "components.json"
    if marker.exists():
        if json.loads(marker.read_text()) != expected or not (prefix / "bin/nvcc").is_file():
            raise RuntimeError(f"Incomplete or different toolchain at {prefix}")
        # Redistributables use lib/, while nvcc's native installation profile
        # searches lib64/. Keep both names for nvcc and CMake consumers.
        if (prefix / "lib").is_dir() and not (prefix / "lib64").exists():
            (prefix / "lib64").symlink_to("lib", target_is_directory=True)
        print(f"Pinned CUDA toolchain already installed: {prefix}")
        return
    if prefix.exists():
        raise FileExistsError(f"Refusing to overwrite {prefix}; choose another --prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    cache = prefix.parent / "downloads"
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cuda-install-", dir=prefix.parent) as temporary:
        staging = Path(temporary) / "toolkit"
        staging.mkdir()
        for name, (relative_url, digest) in COMPONENTS.items():
            archive = cache / Path(relative_url).name
            if not archive.exists() or checksum(archive) != digest:
                partial = archive.with_suffix(archive.suffix + ".partial")
                print(f"Downloading {name}", flush=True)
                with urllib.request.urlopen(BASE_URL + relative_url, timeout=120) as response:
                    with partial.open("wb") as output:
                        shutil.copyfileobj(response, output)
                if checksum(partial) != digest:
                    raise RuntimeError(f"SHA256 mismatch for {name}")
                partial.replace(archive)
            extracted = Path(temporary) / name
            with tarfile.open(archive) as bundle:
                bundle.extractall(extracted, filter="data")
            roots = list(extracted.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise RuntimeError(f"Unexpected archive layout for {name}")
            shutil.copytree(roots[0], staging, dirs_exist_ok=True, symlinks=True)
        if (staging / "lib").is_dir() and not (staging / "lib64").exists():
            (staging / "lib64").symlink_to("lib", target_is_directory=True)
        (staging / "components.json").write_text(json.dumps(expected, indent=2) + "\n")
        staging.rename(prefix)
    print(f"Installed pinned CUDA toolchain: {prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=Path(__file__).resolve().parents[1]
                        / ".toolchains/cuda-12.8.1")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        parser.error("the pinned redistributables target Linux x86_64")
    install(args.prefix.resolve())
