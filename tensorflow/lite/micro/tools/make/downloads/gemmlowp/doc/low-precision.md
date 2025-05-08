# The low-precision paradigm in gemmlowp, and how it's implemented

## Introduction

"Low-precision" means that the input and output matrix entries are integers on
at most 8 bits. The scalar type is uint8_t.

This isn't the same as just doing plain matrix arithmetic over uint8_t, because
that would overflow. To avoid overflow, we internally accumulate results on more
than 8 bits, and at the end we keep only some significant 8 bits. This relies on
the caller providing suitable offset/multiplier/shift parameters, which
effectively govern how we extract some significant 8 bit from our more-than-8bit
temporary accumulators.

## Low-precision paradigms

gemmlowp is flexible enough to support multiple low-precision paradigms, i.e.
multiple ways that a meaning is attached to 8bit values so that a computation
can rely on a 8bit GEMM provided by gemmlowp.

### The current flexible design with arbitrary "output pipelines".

See [output.md](output.md) for more details about output pipelines. This is a
mechanism by which gemmlowp becomes generic enough to support multiple 8bit
computation paradigms, by allowing the user to set up a chain of transformations
to be performed on internal 32bit accumulators to obtain the final outputs.

The public entry point in [public/gemmlowp.h](../public/gemmlowp.h) allowing to
set up an arbitrary output pipeline is `GemmWithOutputPipeline`.

Refer to [quantization.md](quantization.md) for details of how one gets from
first principles to the actual output pipelines to assemble for successful
real-world quantized calculations.

For the scope of the present document, it suffices to say that quantized matrix
multiplication takes the following parameters:

-   The lhs matrix of uint8 quantized values.
-   The rhs matrix of uint8 quantized values.
-   A int32 lhs_offset, that will be added to each entry of the lhs matrix.
-   A int32 rhs_offset, that will be added to each entry of the rhs matrix.
-   An output pipeline, that will process int32 accumulators into final outputs.

The overall computation goes through the following steps:

1.  Cast lhs entries from uint8 to int32 and add lhs_offset to each of them.
2.  Cast rhs entries from uint8 to int32 and add rhs_offset to each of them.
3.  Compute the int32 matrix product of the resulting lhs times rhs.
4.  Apply the output pipeline on these int32 accumulators, to obtain the final
    outputs.

### The legacy low-precision paradigm

This older paradigm is the one exposed by the following entry points:

*   In [public/gemmlowp.h](../public/gemmlowp.h), the `Gemm` entry point.
*   The deprecateed `eight_bit_int_gemm` directory.

Originally, gemmlowp started an implementation of the (now deprecated)
EightBitIntGemm paradigm, where quantized matrix multiplication takes the
following input parameters: - the lhs matrix of uint8 quantized values - the rhs
matrix of uint8 quantized values - the following int32 "quantization
parameters", which control how the uint8 quantized values in the matrices are to
be interpreted during the matrix computation: - lhs_offset - rhs_offset -
result_offset - result_mult_int - result_shift

In that legacy paradigm, the mathematical expression to be computed is the
result of the following steps:

1.  Cast lhs entries from uint8 to int32 and add lhs_offset to each of them.
2.  Cast rhs entries from uint8 to int32 and add rhs_offset to each of them.
3.  Compute the int32 matrix product of the resulting lhs times rhs.
4.  Add result_offset to each entry of the result.
5.  Multiply each entry of the result by the following fraction, and round to
    the nearest integer:

```
result_mult_int
---------------                             (1)
2^result_shift
```

1.  Clamp the resulting int32 values to the `[0..255]` range and cast to uint8.

Again, this paradigm is not recommended for new usage. See
[quantization.md](quantization.md) for how reasoning from first principles, one
arrives to a substantially different quantization paradigm.

In addition, note that the integer multiplication by the numerator in the above
step 5. risks overflowing. That concern is avoided in the currently recommended
output stages by performing a fixed-point multiplication instead of an ordinary
integer multiplication.

# Efficient handling of offsets

At first glance it may seem like the above-described quantized computation
scheme requires adding the lhs_offset and rhs_offset to each of the lhs and rhs
matrix entries.

Doing that in the GEMM kernel would incur substantial overhead: - It would mean
extra arithmetic work in the GEMM kernel; - It would require storing the
lhs_offset and rhs_offset in registers, which would eat into the register space
available for the rest of the GEMM kernel.

One may then consider adding the lhs_offset and rhs_offset once and for all to
lhs and rhs blocks, in a GEMM implementation operating on one lhs block and one
rhs block at a time. However, doing so would require storing lhs and rhs blocks
in 32 bit (or at least in 16 bit in real-world cases), which would partially
negate the memory bandwidth benefits of low-precision computation.

Fortunately, there is another way to handle these offsets that has none of the
costs of the approaches described above. The idea is as follows.

Let `P` denote the matrix shaped like `lhs`, but filled with 1's.

Let `Q` denote the matrix shaped like `rhs`, but filled with 1's.

Adding lhs_offset to each entry of `lhs`, means adding `lhs_offset * P` to
`lhs`.

Adding rhs_offset to each entry of `rhs`, means adding `rhs_offset * Q` to
`rhs`.

Thus, as far as handling `lhs_offset` and `rhs_offset` goes, the matrix product
to be computed is:

```
(lhs + lhs_offset * P) * (rhs + rhs_offset * Q)
```

Expanding this (using distributivity of matrix multiplication over addition), we
see that the above product is equal to the following sum of 4 terms:

```
  lhs * rhs                                 (2)
+ lhs_offset * P * rhs
+ lhs * rhs_offset * Q
+ lhs_offset * rhs_offset * P * Q
```

The first term, `lhs * rhs`, is just the matrix multiplication ignoring the
offsets, i.e. as if `lhs_offset==rhs_offset==0`. Our claim here is that this is
all what we have to compute in the GEMM kernel.

In the second term, `lhs_offset * P * rhs`, notice that since P is filled with
1's, `P * rhs` has all its rows equal to each other, and equal to the row-vector
of sums of all the entries in each column of rhs.

Thus, we can compute the second term, `lhs_offset * P * rhs`, by summing each
column of rhs. This produces a single row-vector, and in order to add the second
term, we simply need to add this row-vector (multiplied by lhs_offset) to each
row of the result. This is just a rank one update of the result (equivalently,
the second term is a rank one matrix), and we can efficiently store it as a
single vector.

The third term, `lhs * rhs_offset * Q`, is entirely similar to the second one,
and can be similarly computed by summing each row of lhs, storing this in a
single column-vector, and later multiplying these sums by rhs_offset.

The fourth term is a single constant, repeated into all the entries of the
matrix. The matrix `P * Q` is filled with the single constant value 'depth' (the
depth of the matrix product i.e. the number of columns of the lhs). Thus the
fourth term is simply the rank zero update adding this constant to each matrix
entry:

```
lhs_offset * rhs_offset * depth
```

# Implementation of this technique in gemmlowp

In gemmlowp, at the packing stage (where we traverse blocks of the lhs and rhs
to prepare them for efficient repeated traversal by the kernel), we compute the
sum of each row of the lhs block and the sum of each column of the rhs block.

See in [internal/pack.h](../internal/pack.h), in the PackedSideBlock class, the
following member:

```
// Handle on the additional buffer backing the vector of sums of slices
// associated with this block. Owned.
Allocator::Handle sums_of_each_slice_handle_;
```

sums_of_each_slice_handle_ is the handle to the buffer allocated to store the
vector containing sums of rows of lhs, or of sums of columns of rhs.

After these rank one updates have been computed at the packing stage, they are
ignored at the compute kernel stage, since that stage is only concerned with the
first of the four terms in (2); they are only used at the unpacking stage. See
the default/reference implementation, `UnpackResultImpl`, in
[internal/unpack.h](../internal/unpack.h).
