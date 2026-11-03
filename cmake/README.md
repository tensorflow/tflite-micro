# TensorFlow Lite for Microcontrollers — CMake Build

Requires **CMake >= 3.23** (3.25+ for `cmake --workflow` presets), a **C++17** compiler (`C` and
`ASM` enabled on demand only when override sources include `.c`/`.S` files), and **Python 3**
(stdlib only for `.tflite`/`.wav`/`.csv`; `Pillow` for `TFLM_BUILD_EXAMPLES=ON`, `numpy` for
`TFLM_BUILD_INTEGRATION_TESTS=ON`).

TFLM (`project(tflm)`) provides a **pure library CMake build** for downstream firmware, RTOS, and
application integration (`add_subdirectory` and `find_package(tflm)`) as well as standalone host
unit testing:
- **Zero network access in CMake**: Dependencies (`flatbuffers`, `kissfft`, and optional vendor
  DSP/NPU SDKs) are resolved strictly from parent CMake targets, local directories (`TFLM_DEPS_DIR`
  / `TFLM_*_DIR`), or `find_package(flatbuffers CONFIG)`.
- **Bring-your-own toolchain & BSP**: Cross-compiled firmware projects supply their own
  `CMAKE_TOOLCHAIN_FILE`, board BSP/linker scripts, and vendor SDK targets (`cmsis-nn`,
  `ethosu_core_driver`, or `TFLM_BACKEND_TARGETS`), while TFLM specializes kernel and platform
  sources without mutating global flags or forcing `CXX_EXTENSIONS OFF`.
- **Direct source & test discovery**: Library sources, `FILE_SET HEADERS`, and CTest targets are
  discovered directly by `cmake/tflm_sources.cmake` (`CONFIGURE_DEPENDS`) without a separate codegen
  step.

## Quick Start (Standalone Host Build & CI Tests)

Populate a local dependencies directory once via `cmake/download_deps.py` (pinned to `MODULE.bazel`
SHA-256 digests), then run any preset from `CMakePresets.json`:

```bash
export TFLM_DEPS_DIR="$PWD/build/deps"
python3 cmake/download_deps.py "$TFLM_DEPS_DIR"
cmake --workflow --preset default
```

Presets: `default` (`RelWithDebInfo` + unit/example tests), `debug` (`Debug` + `-Werror` + tests),
`core-only` (`MinSizeRel` `tflm::micro` without `tflm::signal` or KissFFT), and `ci`
(`RelWithDebInfo` + `-Werror` + unit/example tests).

To run the full out-of-tree CI test suite (downloading dependencies into a temporary directory so
the repository stays clean):

```bash
tensorflow/lite/micro/tools/ci_build/test_cmake.sh           # runs ci, compression_debug, core_only, interop
tensorflow/lite/micro/tools/ci_build/test_cmake.sh ci        # single mode
```

## Directory Layout

```text
CMakeLists.txt                       Options, local deps, targets (tflm::micro, tflm::signal), CTest & install
CMakePresets.json                    Host configure, build, test, and workflow presets
cmake/
  README.md                          CMake build documentation
  download_deps.py                   Standalone FlatBuffers & KissFFT fetcher for host testing/CI
  tflm_codegen.cmake                 Public tflm_target_embed_assets() model/asset codegen module
  tflm_config.cmake.in               Relocatable find_package(tflm) config template
  tflm_sources.cmake                 Source, FILE_SET HEADERS, and CTest target discovery
  tests/                             Consumer project (CMakeLists.txt, consumer_main.cc) for package smoke tests
tensorflow/lite/micro/tools/ci_build/
  test_cmake.sh                      Out-of-tree CI build, CTest, and package/interop runner
```

## Public Targets & Helpers

- `tflm::micro` — Core interpreter, arena allocator, public headers (`FILE_SET HEADERS`), and
  reference or specialized kernels.
- `tflm::signal` — Audio/DSP signal kernels (`TFLM_ENABLE_SIGNAL=ON`); KissFFT is linked `PRIVATE`
  and never leaked into consumer include paths or exports.
- `tflm::test_support` — Unit test harness (`KernelRunner`, `testing::` mocks); always defined
  (`EXCLUDE_FROM_ALL`) for TFLM and downstream custom-op tests.
- `tflm_target_embed_assets(<target> <PRIVATE|PUBLIC> [FILES] <asset1> ...)` — Converts
  `.tflite`, `.wav`, `.bmp`, `.csv`, and `.npy` files into deduplicated `alignas(16)` C++ `OBJECT`
  libraries and attaches them onto `<target>` (in both `add_subdirectory` and `find_package(tflm)`).

## Consuming TFLM in Downstream Projects

### 1. Portable `add_subdirectory` (Host, Zephyr, Pico SDK, ESP-IDF, Generic MCU)

`TFLM_BUILD_TESTS`, `TFLM_ENABLE_SIGNAL`, and `TFLM_INSTALL` default to `OFF` when consumed as a
subproject (`zephyr_interface` is linked automatically when present):

```cmake
add_subdirectory(third_party/tflite-micro)
target_link_libraries(my_firmware PRIVATE tflm::micro)
tflm_target_embed_assets(my_firmware PRIVATE models/keyword_model.tflite)
```

> **FlatBuffers version mismatch (`static_assert`):** Checked-in `*_generated.h` headers assert
> FlatBuffers `25.9.23`. If your project supplies a different FlatBuffers version, pass
> `-DTFLM_FLATC_EXECUTABLE=/path/to/flatc` matching your FlatBuffers headers to regenerate the
> schemas at build time.

### 2. Firmware with Hardware-Optimized Kernels (`CMSIS-NN`, `Ethos-U`, `Xtensa`, `Hexagon`, Custom NPU)

Supply your firmware's `CMAKE_TOOLCHAIN_FILE` (setting target CPU/ABI flags such as `-mcpu`,
`-mfloat-abi`, or DSP core flags) and vendor SDK target(s) in the parent project, then configure
TFLM's specialization. For Xtensa, Hexagon, or custom SDKs, pass parent target(s) via
`TFLM_BACKEND_TARGETS` (Xtensa targets should define `-DHIFI3`, `-DHIFI4`, `-DHIFI5`, or
`-DVISION_P6` via `TFLM_BACKEND_TARGETS`):

```cmake
add_subdirectory(third_party/cmsis-nn)             # defines `cmsis-nn`
add_subdirectory(third_party/ethos-u-core-driver)  # defines `ethosu_core_driver`
set(TFLM_OPTIMIZED_KERNELS cmsis_nn)
set(TFLM_CO_PROCESSOR ethos_u)
set(TFLM_SYSTEM_TARGET cortex_m_generic)
add_subdirectory(third_party/tflite-micro)
target_link_libraries(my_firmware PRIVATE tflm::micro)
```

Firmware builds using `-Os` (`MinSizeRel`) globally can selectively compile TFLM kernels at `-O2`:

```cmake
get_target_property(_srcs tflm_micro SOURCES)
list(FILTER _srcs INCLUDE REGEX "/kernels/")
set_source_files_properties(${_srcs} TARGET_DIRECTORY tflm_micro
  PROPERTIES COMPILE_OPTIONS "-O2")
```

### 3. `cmake --install` and `find_package(tflm)`

Parent projects enabling `TFLM_INSTALL=ON` while linking in-tree dependency targets (`flatbuffers`,
`cmsis-nn`, `TFLM_BACKEND_TARGETS`, etc.) must also `install(EXPORT ...)` those targets.

```cmake
# After: cmake --install build/default --prefix /tmp/tflm-install
find_package(tflm CONFIG REQUIRED COMPONENTS micro signal)
add_executable(my_app main.cc)
target_link_libraries(my_app PRIVATE tflm::micro tflm::signal)
tflm_target_embed_assets(my_app PRIVATE models/keyword_model.tflite)
```

## Key Build Options

| Option | Default | Description |
|---|---|---|
| `TFLM_DEPS_DIR` | `"$ENV{TFLM_DEPS_DIR}"` | Optional base directory containing local `flatbuffers` / `kissfft` |
| `TFLM_FLATBUFFERS_DIR` / `TFLM_KISSFFT_DIR` | `""` | Explicit local paths to FlatBuffers or KissFFT |
| `TFLM_FLATC_EXECUTABLE` | `""` | Explicit `flatc` binary to regenerate `*_generated.h` when FlatBuffers `!= 25.9.23` |
| `TFLM_ENABLE_SIGNAL` | `${PROJECT_IS_TOP_LEVEL}` | Build `tflm::signal` (requires local KissFFT >= 131.1.0 or target `tflm::kissfft`) |
| `TFLM_ENABLE_COMPRESSION` | `OFF` | Enable LUT weight decompression (`reference` and `xtensa` kernels) |
| `TFLM_STRIP_ERROR_STRINGS` | `OFF` | Define `TF_LITE_STRIP_ERROR_STRINGS` to reduce binary size |
| `TFLM_EXCLUDE_SYSTEM_IMPL` | `OFF` | Omit default `debug_log`/`micro_time`/`system_setup` sources |
| `TFLM_REMOVE_VIRTUAL_DELETE` | `${CMAKE_CROSSCOMPILING}` | Define `TFLM_REMOVE_VIRTUAL_DELETE` to omit virtual `operator delete` |
| `TFLM_WARNINGS_AS_ERRORS` | `OFF` | Treat compiler warnings as errors (`-Werror` / `/WX`) |
| `TFLM_LINK_LIBM` | `ON` (`OFF` on `ARMClang`, `MSVC`, Zephyr) | Link `-lm` publicly on `tflm::micro` (disable for `-nostdlib` builds) |
| `TFLM_OPTIMIZED_KERNELS` | `""` | Specialized kernel backend (`cmsis_nn`, `xtensa`, `hexagon`) |
| `TFLM_CO_PROCESSOR` | `""` | Co-processor backend (`ethos_u`) |
| `TFLM_SYSTEM_TARGET` | `""` | Platform override directory under `tensorflow/lite/micro/` (e.g., `cortex_m_generic`) |
| `TFLM_BACKEND_TARGETS` | `""` | Semicolon-separated parent CMake target(s) providing vendor DSP/NPU headers, defines, & libs |
| `TFLM_EXTERNAL_KERNEL_DIR` | `""` | Semicolon-separated out-of-tree directory(ies) of custom/vendor kernel overrides |
| `TFLM_OPTIMIZE_KERNELS_FOR` | `"SPEED"` | Kernel scratch-buffer tradeoff (`SPEED` or `SIZE`) |
| `TFLM_BUILD_TESTS` / `TFLM_BUILD_EXAMPLES` | `${PROJECT_IS_TOP_LEVEL}` | Build unit and example test targets |
| `TFLM_BUILD_INTEGRATION_TESTS` | `OFF` | Build SeaNet integration tests (requires `TFLM_ENABLE_SIGNAL=ON`) |
| `TFLM_EXCLUDED_TESTS` | `""` | Semicolon-separated list of test target names to exclude |
| `TFLM_INSTALL` | `${PROJECT_IS_TOP_LEVEL}` | Enable `cmake --install` (`FILE_SET HEADERS`) and `tflm-config.cmake` export |
