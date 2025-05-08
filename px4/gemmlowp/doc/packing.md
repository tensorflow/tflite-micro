# The packing stage in gemmlowp

## Introduction

We assume familiarity with [design.md](design.md) and with the overall 3 stages
of computations described there: packing, kernel, unpacking.

This page goes into more details about the first stage: packing.

We also assume familiarity with [kernel.md](kernel.md) as it describes the
packed format requirements that the kernels expect, and that forms basically the
contract that the packing stage must honor.

Some parts below also assume familiarity with
[low-precision.md](low-precision.md) as the packing stage also has to compute
the vectors of sums or columns as described there.

## The storage order of packed blocks, partly hidden behind sequential access

As explained in [design.md](design.md), the primary purpose of packing is to
ensure that when the kernel traverses Lhs/Rhs matrix data, it can do so
efficiently thanks to having the data stored in an order that is as similar as
possible to the order in which the compute stage has to traverse this data.

This traversal order is nontrivial for the reasons outlined in
[design.md](design.md): at the innermost level, one tries to work within
registers as much as possible; at the next level, one tries to stay within L1
cache as much as possible. The packed blocks that we handle are supposed to fit
entirely in L2 cache.

Thus it has become standard in GEMM design to describe complicated "Z-order" or
"fractal order" storage for packed blocks.

However, we should keep in mind that the whole point of the packed storage order
is to be as similar as possible to the order of traversal during the compute
stage. The storage order doesn't matter in itself; the only thing that matters
is simple access patterns during the compute stage.

This suggests the following approach to implementing packing: take the exact
same hierarchy of nested loops of the compute stage, drop the loops that are not
relevant to the side (Lhs or Rhs) being packed, and try to use mostly sequential
access to the destination packed data.

This hierarchy of nested loops can be seen in PackSideBlockImpl (PackL2, PackL1,
PackRun), compare to the similar hierarchy of loops in internal/compute.h.

In this way, the more intricate small-scale details or the packed data format
never need to be made explicit (which would be complicated). We still use some
"seeking" but only at larger scales, where the storage order is less \
complicated to describe.

### Sequential access to PackedSideBlock data

See PackedSideBlock in internal/pack.h, specifically the following data members:

```
// Handle on the buffer backing this packed block. Owned.
Allocator::Handle data_handle_;
```

and:

```
// pos_ is the current position in the buffer, which we access
// sequentially, like a file.
// The idea is that we pack data in the same order as it is
// going to be traversed during the computation, which for
// cache-friendliness reasons is complicated to random-access,
// as the offsets calculations would be intricate. So we
// give up random-access addressing, and instead content ourselves
// with sequential access.
//
// pos_ is mutable because during the computation we will want to
// be able to iterate on the data in a const PackedSideBlock.
mutable int pos_;
```

The methods exposing sequential access are:

```
std::uint8_t* current_data() {
  return allocator_->GetPointer<std::uint8_t>(data_handle_) + pos_;
}
```

and:

```
void seek_next_cell() const { pos_ += KernelSideFormat::Cell::kSize; }

void seek_forward_n_cells(int n) const {
  pos_ += n * KernelSideFormat::Cell::kSize;
}
```

### Random access to PackedSideBlock data at larger scales

We still need some random access at larger scales (with high granularity), which
is unavoidable since GEMM is O(n^3) and has to traverse each of the O(n^2)
inputs O(n) times.

The watershed between sequential access and random access is at the level of a
'Run'. Throughout gemmlowp we consistently use the term 'Run' to refer to the
innermost GEMM loop in the depth dimension. That's the critical inner loop that
must be as fast as possible, thus for which we absolutely want sequential access
to packed data so that the storage order is optimal by construction. At larger
scales i.e. between runs, we accept that the storage order is less optimal and
since it's also less intricate, it's not too hard to implement random access
there.

This is done by the seek_run method:

```
void seek_run(int start_width, int start_depth) const {
  int kernel_run_depth =
      std::min<int>(params_.l1_depth, params_.l2_depth - start_depth);
  pos_ = params_.l2_width * start_depth + start_width * kernel_run_depth;
}
```

We see that the formula involves the l1_depth parameter, which is how the packed
storage order depends on L1 cache size. Again, the whole packed block is
supposed to fit in L2 cache.

## The innermost loop of the packing stage, PackRun, and PackingRegisterBlock

Keeping with our consistent usage of the term 'Run' throughout gemmlowp, the
innermost loop is called PackRun().

Here we recall a very important principle that was explained in
[kernels.md](kernels.md): the kernel is free to dictate the precise data format
that it expects; the packing code has to honor it. So there's an asymmetry here:
the kernel is the master, the packing is the slave. That's why the packing code
is templatized in the KernelSideFormat. At larger scales, the packing is
independent of kernel format details, but inside PackRun is where we take care
of the small-scale details that do depend on the kernel format details. That's
why it's a good thing that we only need sequential access here, as it would be
very complicated to spell out random access at this scale.

Anyway, PackRun.

Since it is the critical inner loop, it is what we want to allow specializing
for particular CPU architectures. To allow that, we handle at a time blocks of
fixed dimensions, that is intended to be friendly enough to optimization. These
blocks are PackingRegisterBlock's and their dimensions are:

```
  width = KernelWidth
  depth = kRegisterSize
```

See [kernels.md](kernels.md) and internal/kernel.h for the former, and
internal/common.h for the latter.

See the comments around PackingRegisterBlock in internal/pack.h:

```
// A PackingRegisterBlock is a small fixed-size block of a matrix being
// packed. This class is the generic non-optimized implementation,
// it is inherited by the generic implementation of PackingRegisterBlock,
// which may be overriden by template specialization. Overriding it is how
// one may provide optimized packing code paths.
//
// The packing of a block proceeds in two steps:
//   1. Ensuring that we have a complete block of source data, i.e. a block of
//      the compile-time prescribed size. This is where we handle unaligned
//      boundaries: if we don't have a complete block of source data, then
//      we copy and zero-extend it into a local temporary (complete_src_),
//      see MakeCompleteSrc. In the generic case, we do have a complete block,
//      so we just use it in-place, see UseCompleteSrcInPlace.
//   2. Packing a complete block into the destination, see Pack. This is the
//      most critical part, so it's convenient that unaligned boundaries have
//      already been handled in step 1.
```

## Other things that the packing stage has to do

Besides storing matrix entries in a suitable order, the packing stages also has
two other things to do.

First, packing has to compute the vectors of sums of entries along the depth
dimension. If this is any mysterious, read [low-precision.md](low-precision.md).
These will only be used at the unpacking stage.

Second, if the BitDepthSetting requires less than 8 bit of precision, then at
the packing stage we have to requantize inputs accordingly. See
[less-than-8-bit.md](less-than-8-bit.md) for details. This is the Requantize()
function.

## Specialized packing paths for specific formats on specific CPU architectures

Please refer to internal/pack_neon.h for examples of doing that. The piece of
code to be specialized is PackingRegisterBlock. However, inside of it, only the
Pack() method typically needs to be specialized (the rest is unlikely to be
critical). So one typically specializes PackingRegisterBlock but still
inheriting PackingRegisterBlockBase to keep the generic stuff, and then one
typically wants to override the Pack() method.

Template specialization for the right template parameters is how one specifies
in which case a given path is to be used in place of the generic packing code.

It is entirely possible to set the value of kRegisterSize differently based on
the CPU architecture (for example, 32 on x86 with AVX) as long as all the
specialized packing paths used on that CPU architecture are consistent with it.
