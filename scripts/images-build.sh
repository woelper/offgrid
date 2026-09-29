#!/usr/bin/env bash
# Build the pieces `cargo` cannot, then hand off to cargo with its arguments.
#
#     ./scripts/images-build.sh run --release --features images
#
# llama.cpp and stable-diffusion.cpp both embed ggml, and two static copies in
# one binary collide on every symbol. Both can consume an external one instead,
# so this builds ggml once and points each at it. The catch is GGML_MAX_NAME:
# it sits inside ggml_tensor, sd.cpp static_asserts it is at least 160, and
# upstream defaults to 64 — every object that sees ggml.h must agree, or they
# disagree about how big a tensor is. It is behind #ifndef, so a define is all
# it takes.
set -euo pipefail
cd "$(dirname "$0")/.."

# Deliberately not under target/: Swatinem/rust-cache prunes that directory
# between runs, and it restored a stable-diffusion.cpp checkout with its
# contents stripped — which the old "does the directory exist" guard took for a
# working one, so the fetch was skipped and cmake found no CMakeLists.txt.
SD_ROOT="$PWD/.sd-cpp"
PREFIX="$SD_ROOT/prefix"
DEF="-DGGML_MAX_NAME=160"
JOBS="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# cmake wants native paths. Under git-bash on Windows this script sees
# /d/a/offgrid/... , which cmake.exe cannot open; cygpath turns that back into
# a drive letter. Everywhere else it is the identity.
cmpath() {
    if command -v cygpath > /dev/null 2>&1; then
        cygpath -m "$1"
    else
        printf '%s' "$1"
    fi
}

# The shared ggml keeps whatever GPU backend the platform has, or an images
# build would take acceleration away from chat in the same binary too — one
# ggml serves both, so a CPU-only ggml means a CPU-only everything. sd.cpp and
# llama.cpp both pick the backend up from the ggml they link against, so
# neither needs a flag of its own.
#
# OFFGRID_IMAGES_GPU forces the choice: `none` for a CPU-only build, `vulkan`
# to insist (and fail loudly if the SDK is missing) rather than quietly
# falling back. The default detects.
GGML_GPU=()
GPU_TAG=cpu
WANT="${OFFGRID_IMAGES_GPU:-auto}"
if [[ "$(uname -s 2>/dev/null)" == "Darwin" ]]; then
    if [[ "$WANT" != "none" ]]; then
        # BLAS off for the same reason llama-cpp-sys-2's own build script turns
        # it off on Apple: ggml would enable it by default and the trimmed
        # llama.cpp the crate vendors has no ggml-blas directory to build.
        GGML_GPU=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=OFF)
        GPU_TAG=metal
    fi
elif [[ "$WANT" != "none" ]]; then
    # Vulkan rather than CUDA: one backend for every vendor, and the only one
    # that can be built on a runner without a card in it. glslc compiles the
    # shaders and is the part that is actually missing when it is missing;
    # VULKAN_SDK is how the Windows SDK announces itself.
    if command -v glslc > /dev/null 2>&1 || [[ -n "${VULKAN_SDK:-}" ]]; then
        GGML_GPU=(-DGGML_VULKAN=ON)
        GPU_TAG=vulkan
    elif [[ "$WANT" == "vulkan" ]]; then
        echo "error: OFFGRID_IMAGES_GPU=vulkan but no glslc and no VULKAN_SDK" >&2
        exit 1
    fi
fi
# Under git-bash on Windows, cmake with the Ninja generator takes the first
# compiler on PATH, and that is MinGW — which builds ggml into .a archives the
# MSVC-targeting cargo build cannot link, and a vulkan-shaders-gen.exe that
# dies at startup (0xc0000139) for want of its runtime DLLs. cmake-rs passes
# the compiler explicitly for exactly this reason; calling cmake ourselves, we
# have to. The Visual Studio generator would pick cl on its own, but ggml's
# shader sub-build races under MSBuild, so Ninja it is.
CC_ARGS=()
if command -v cl > /dev/null 2>&1; then
    CC_ARGS=(-DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl)
fi

echo "==> ggml backend: $GPU_TAG"

# A prefix built for one backend is wrong for another, and the guards below
# only ask whether the libraries exist. Without this, adding the Vulkan SDK to
# a tree that has already been built once gets you a silently CPU-only binary.
STAMP="$PREFIX/.offgrid-ggml-backend"
if [[ -f "$STAMP" && "$(cat "$STAMP")" != "$GPU_TAG" ]]; then
    echo "==> backend changed ($(cat "$STAMP") -> $GPU_TAG), rebuilding ggml and sd.cpp"
    rm -rf "$PREFIX" "$SD_ROOT/ggml-build" "$SD_ROOT/sd-build"
fi

# llama.cpp's own ggml is the one to share: sd.cpp has an upstream-ggml mode
# for exactly this, while llama.cpp has no mode for sd.cpp's patched fork.
# Windows runners ship `python`, not always `python3`.
PY="$(command -v python3 || command -v python)"
if [[ -z "$PY" ]]; then
    echo "error: need python to locate llama-cpp-sys-2's sources" >&2
    exit 1
fi

llama_src() {
    cargo metadata --format-version 1 --no-deps > /dev/null 2>&1 || true
    cargo metadata --format-version 1 2>/dev/null | "$PY" -c '
import json, sys, pathlib
meta = json.load(sys.stdin)
for pkg in meta["packages"]:
    if pkg["name"] == "llama-cpp-sys-2":
        print(pathlib.Path(pkg["manifest_path"]).parent / "llama.cpp")
        break
'
}

LLAMA="$(llama_src)"
if [[ -z "$LLAMA" || ! -d "$LLAMA" ]]; then
    echo "error: could not find llama-cpp-sys-2's vendored llama.cpp" >&2
    exit 1
fi

# stable-diffusion.cpp: a checkout you provide, or one fetched next to the
# build. Nothing here is vendored into the repo yet — that is a decision for
# when this stops being a prototype.
SD_SRC="${OFFGRID_SD_SRC:-$SD_ROOT/src}"
# Pinned, not master: this is the revision the model list was tested against,
# and sd.cpp moves fast enough that an unpinned build would be a different
# program every week. Bump it deliberately, after running the models.
SD_REF="${OFFGRID_SD_REF:-c92d73c408515c94beef32161bb5960764fde7a0}"
if [[ ! -f "$SD_SRC/CMakeLists.txt" ]]; then
    echo "==> fetching stable-diffusion.cpp $SD_REF into $SD_SRC"
    # A directory without the sources is worse than no directory: whatever is
    # there is a remnant, not a checkout.
    rm -rf "$SD_SRC"
    mkdir -p "$SD_ROOT"
    git init -q "$SD_SRC"
    git -C "$SD_SRC" remote add origin https://github.com/leejet/stable-diffusion.cpp
    git -C "$SD_SRC" fetch -q --depth 1 origin "$SD_REF"
    git -C "$SD_SRC" checkout -q FETCH_HEAD
fi

# The marker differs by toolchain, as does everything else about a static
# library's name.
if ! ls "$PREFIX"/lib/libggml-base.a "$PREFIX"/lib/ggml-base.lib > /dev/null 2>&1; then
    echo "==> building ggml (shared by llama.cpp and stable-diffusion.cpp)"
    cmake -S "$(cmpath "$LLAMA")" -B "$(cmpath "$SD_ROOT/ggml-build")" \
        ${GGML_GPU[@]+"${GGML_GPU[@]}"} \
        ${CC_ARGS[@]+"${CC_ARGS[@]}"} \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$(cmpath "$PREFIX")" \
        -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_C_FLAGS="$DEF" -DCMAKE_CXX_FLAGS="$DEF" \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_COMMON=OFF -DLLAMA_CURL=OFF
    cmake --build "$SD_ROOT/ggml-build" --config Release -j"$JOBS" --target install
fi

if ! ls "$PREFIX"/lib/libstable-diffusion.a "$PREFIX"/lib/stable-diffusion.lib > /dev/null 2>&1; then
    echo "==> building stable-diffusion.cpp against that ggml"
    cmake -S "$(cmpath "$SD_SRC")" -B "$(cmpath "$SD_ROOT/sd-build")" \
        ${CC_ARGS[@]+"${CC_ARGS[@]}"} \
        -DCMAKE_BUILD_TYPE=Release \
        -DSD_USE_SYSTEM_GGML=ON -DSD_USE_UPSTREAM_GGML=ON \
        -DSD_GGML_SOURCE_DIR="$(cmpath "$LLAMA/ggml")" -DSD_BUILD_EXAMPLES=OFF \
        -DCMAKE_C_FLAGS="$DEF" -DCMAKE_CXX_FLAGS="$DEF" \
        -DCMAKE_PREFIX_PATH="$(cmpath "$PREFIX")"
    cmake --build "$SD_ROOT/sd-build" --config Release -j"$JOBS"
    # libstable-diffusion.a on unix, stable-diffusion.lib under MSVC, and the
    # latter lands in a per-configuration subdirectory.
    SD_LIB="$(find "$SD_ROOT/sd-build" \( -name "libstable-diffusion.a" -o -name "stable-diffusion.lib" \) | head -1)"
    if [[ -z "$SD_LIB" ]]; then
        echo "error: stable-diffusion.cpp built but produced no static library" >&2
        exit 1
    fi
    cp "$SD_LIB" "$PREFIX/lib/"
    cp "$SD_SRC/include/stable-diffusion.h" "$PREFIX/include/"
fi

mkdir -p "$PREFIX"
printf '%s' "$GPU_TAG" > "$STAMP"

echo "==> cargo $*"
# CFLAGS/CXXFLAGS reach llama.cpp's own build through the cc crate, so its
# ggml.h sees the same GGML_MAX_NAME as the library it will link against.
export OFFGRID_SD_PREFIX="$PREFIX"
export CMAKE_PREFIX_PATH="$PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CFLAGS="$DEF ${CFLAGS:-}"
export CXXFLAGS="$DEF ${CXXFLAGS:-}"
exec cargo "$@"
