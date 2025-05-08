# gemmlowp: a small self-contained low-precision GEMM library

[![Build Status](https://secure.travis-ci.org/google/gemmlowp.png)](http://travis-ci.org/google/gemmlowp)

This is not a full linear algebra library, only a GEMM library: it only does
general matrix multiplication ("GEMM").

The meaning of "low precision" is detailed in this document:
[doc/low-precision.md](doc/low-precision.md)

Some of the general design is explained in [doc/design.md](doc/design.md).

**Warning:** This library goes very slow if compiled incorrectly; see below.

## Disclaimer

This is not an official Google product (experimental or otherwise), it is just
code that happens to be owned by Google.

## Mailing list

gemmlowp-related discussion, about either development or usage, is welcome on
this Google Group (mailing list / forum):

https://groups.google.com/forum/#!forum/gemmlowp

## Portability, target platforms/architectures

Should be portable to any platform with some C++11 and POSIX support, while we
have optional optimized code paths for specific architectures.

Required:

*   C++11 (a small conservative subset of it)

Required for some features:

*   Some POSIX interfaces:
    *   pthreads (for multi-threaded operation and for profiling).
    *   sysconf (for multi-threaded operation to detect number of cores; may be
        bypassed).

Optional:

*   Architecture-specific code paths use intrinsics or inline assembly. See
    "Architecture-specific optimized code paths" below.

## Architecture-specific optimized code paths

We have some optimized code paths for specific instruction sets. Some are
written in inline assembly, some are written in C++ using intrinsics. Both GCC
and Clang are supported.

Current optimized code paths:

*   ARM with NEON (both 32bit and 64bit).
*   Intel x86 with SSE 4.1 (both 32bit and 64bit).

When building for x86, it's very important to pass `-msse4.1` to the compiler,
otherwise gemmlowp will use slow reference code. Bazel users can compile by
running `bazel build --copt=-msse4.1 //gemmlowp:all`. The compiled binary should
work on all Intel CPUs since 2008 (including low power microarchitectures) as
well as AMD CPUs since 2011.

Please note when compiling binaries that don't need to be distributed, it's
generally a better idea to pass `-march=native` to the compiler. That flag
implies `-msse4.1` flag, along with others that might be helpful. This of course
assumes the host machine supports those instructions. Bazel users should prefer
to run `bazel build --config=opt //gemmlowp:all` instead.

Details of what it takes to make an efficient port of gemmlowp, namely writing a
suitable GEMM kernel and accompanying packing code, are explained in this file:
[doc/kernel.md](doc/kernel.md).

## Public interfaces

### The gemmlowp public interface

gemmlowp's main public interface is in the `public/` subdirectory.

This is a headers-only library, so there is nothing to link to.

Usage documentation, and comments on the deprecation status of each public entry
point, may be found in [doc/public.md](doc/public.md) .

A full, self-contained usage example, showing how to quantize float matrices and
perform a quantized matrix multiplication approximating a float matrix
multiplication, is given in
[doc/quantization_example.cc](doc/quantization_example.cc).

### Old EightBitIntGemm legacy deprecated interface

The `eight_bit_int_gemm/` subdirectory contains an alternate interface that
should be considered purely legacy, deprecated, and going to be removed at some
point in the future.

## Building

### Building by manually invoking your compiler

Because gemmlowp is so simple, working with it involves only single-command-line
compiler invocations. Therefore we expect that most people working with gemmlowp
will either manually invoke their compiler, or write their own rules for their
own preferred build system.

Keep in mind (previous section) that gemmlowp itself is a pure-headers-only
library so there is nothing to build.

For a Android gemmlowp development workflow, the `scripts/` directory contains a
script to build and run a program on an Android device:

```
scripts/test-android.sh
```

### Building using Bazel

That being said, we also maintain a Bazel BUILD system as part of gemmlowp. Its
usage is not mandatory at all and is only one possible way that gemmlowp
libraries and tests may be built. If you are interested, Bazel's home page is
http://bazel.build/ And you can get started with using Bazel to build gemmlowp
targets by first creating an empty WORKSPACE file in a parent directory, for
instance:

```
$ cd gemmlowp/..  # change to parent directory containing gemmlowp/
$ touch WORKSPACE # declare that to be our workspace root
$ bazel build gemmlowp:all
```

## Testing

### Testing by manually building and running tests

The test/ directory contains unit tests. The primary unit test is

```
test/test.cc
```

Since it covers also the EightBitIntGemm interface, it needs to be linked
against

```
eight_bit_int_gemm/eight_bit_int_gemm.cc
```

It also uses realistic data captured from a neural network run in

```
test/test_data.cc
```

Thus you'll want to pass the following list of source files to your
compiler/linker:

```
test/test.cc
eight_bit_int_gemm/eight_bit_int_gemm.cc
test/test_data.cc
```

The `scripts/` directory contains a script to build and run a program on an
Android device:

```
scripts/test-android.sh
```

It expects the `CXX` environment variable to point to an Android toolchain's C++
compiler, and expects source files (and optionally, cflags) as command-line
parameters. To build and run the above-mentioned main unit test, first set `CXX`
e.g.:

```
$ export CXX=/some/toolchains/arm-linux-androideabi-4.8/bin/arm-linux-androideabi-g++
```

Then run:

```
$ ./scripts/test-android.sh \
test/test.cc \
eight_bit_int_gemm/eight_bit_int_gemm.cc \
test/test_data.cc
```

### Testing using Bazel

Alternatively, you can use Bazel to build and run tests. See the Bazel
instruction in the above section on building. Once your Bazel workspace is set
up, you can for instance do:

```
$ bazel test gemmlowp:all
```

## Troubleshooting Compilation

If you're having trouble finding the compiler, follow these instructions to
build a standalone toolchain:
https://developer.android.com/ndk/guides/standalone_toolchain.html

Here's an example of setting up Clang 3.5:

```
$ export INSTALL_DIR=~/toolchains/clang-21-stl-gnu
$ $NDK/build/tools/make-standalone-toolchain.sh \
--toolchain=arm-linux-androideabi-clang3.5 --platform=android-21 \
--install-dir=$INSTALL_DIR
$ export CXX="$INSTALL_DIR/bin/arm-linux-androideabi-g++ \
--sysroot=$INSTALL_DIR/sysroot"
```

Some compilers (e.g. the default clang++ in the same bin directory) don't
support NEON assembly. The benchmark build process will issue a warning if
support isn't detected, and you should make sure you're using a compiler like
arm-linux-androideabi-g++ that does include NEON.

## Benchmarking

The main benchmark is

```
test/benchmark.cc
```

It doesn't need to be linked to any other source file. We recommend building
with assertions disabled (`-DNDEBUG`).

For example, the benchmark can be built and run on an Android device by doing:

```
$ ./scripts/test-android.sh test/benchmark.cc -DNDEBUG
```

If `GEMMLOWP_TEST_PROFILE` is defined then the benchmark will be built with
profiling instrumentation (which makes it slower) and will dump profiles. See
next section on profiling.

## Profiling

The `profiling/` subdirectory offers a very simple, naive, inaccurate,
non-interrupting sampling profiler that only requires pthreads (no signals).

It relies on source code being instrumented with pseudo-stack labels. See
`profiling/instrumentation.h`. A full example of using this profiler is given in
the top comment of `profiling/profiler.h`.

## Contributing

Contribution-related discussion is always welcome on the gemmlowp mailing list
(see above).

We try to keep a current list of TODO items in the `todo/` directory.
Prospective contributors are welcome to pick one to work on, and communicate
about it on the gemmlowp mailing list.

Details of the contributing process, including legalese, are in CONTRIBUTING.

## Performance goals

Our performance goals differ from typical GEMM performance goals in the
following ways:

1.  We care not only about speed, but also about minimizing power usage. We
    specifically care about charge usage in mobile/embedded devices. This
    implies that we care doubly about minimizing memory bandwidth usage: we care
    about it, like any GEMM, because of the impact on speed, and we also care
    about it because it is a key factor of power usage.

2.  Most GEMMs are optimized primarily for large dense matrix sizes (>= 1000).
    We do care about large sizes, but we also care specifically about the
    typically smaller matrix sizes encountered in various mobile applications.
    This means that we have to optimize for all sizes, not just for large enough
    sizes.
