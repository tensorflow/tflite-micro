# Computation with less than 8 bits in gemmlowp

## Introduction

We assume familiarity with gemmlowp's low-precision uint8 computation paradigm,
which is described in [low-precision.md](low-precision.md).

This document is about the possibility of further reducing precision below 8
bits.

That allows to get higher arithmetic throughput on some architectures, at the
cost of decreased accuracy.

## The past, present, and future of less-than-8-bit computation in gemmlowp

A meta note is needed here as to how this fits with the general gemmlowp design.

### The past

Less-than-8-bit computation was initially designed and implemented in gemmlowp
as a drop-in replacement for regular 8bit computation, a plain optimization. The
idea was that to automatically requantize 8bit operands to less-than-8bit during
the O(N^2) packing stage, then take advantage of the lower bit depth during the
O(N^3) compute stage. For large enough matrices, that should be worth it.

### The present

TODO(benoitjacob): update this documentation. This 'present' state just
became the past (February 2017).

At the moment, this less-than-8-bit mode of gemmlowp is not much used in
practice, because the implicit requantization of operands from 8bit to
less-than-8bit turned out to be more expensive than initially expected, both in
terms of speed and accuracy:

1.  Speed: the O(N^2) requantization is only negligible compared to the O(N^3)
    compute kernel when the matrix size N is large enough; in practice, smaller
    matrix sizes turned out to be very important, making the requantization
    approach slower than expected.

2.  Accuracy: As neural networks were optimized for size, their sensitivity to
    numerical accuracy increased. Then the approach of requantizing
    already-quantized data turned out to be more wasteful of accuracy than we
    could afford.

### The future

Less-than-8bit still probably has good prospects; what should be dropped is the
requantization. In other words, in the future, we might have neural networkds
trained right away for some bit depth lower than 8 bits. The resulting values
would probably still be stored as 8 bits (unless the bit depth eventually
becomes very low). Thus, no particular work would be needed in the packing
stage; no overhead or loss of accuracy would be incurred anymore.

In other words: the design of less-than-8-bit kernels is probably useful in the
long run; what is on the way out is requantization and packing/unpacking-stage
aspects.

With that said, the rest of this page retains its old content about the present
approach:

## Public interface

### The BitDepthSetting parameter in the EightBitIntGemm interface

Accessing less-than-8-bit computation via the EightBitIntGemm is very simple:
EightBitIntGemm takes a BitDepthSetting enum which allows to choose among a
fixed set of supported bit-depth combinations.

### The BitDepthParams parameter in the public/gemmlowp.h interface

The public/gemmlowp.h interface exposes more extensive control over
quantization, by means of a BitDepthParams template parameter, which is a type
parameter, carrying information about: 1. The LHS and RHS bit depth, which can
be set arbitrarily and independently; 2. The 'RoundingStrategy', which is the
heuristic used to choose a rounding mode, based on the accumulation size (a.k.a.
the "depth" dimension of the Gemm). Details can be seen in public/bit_depth.h.

### How does BitDepth{Setting,Params} affect input/output uint8 matrix data?

Input/output matrix data is all uint8's, ranging from 0 to 255, regardless of
the BitDepth{Setting,Params}.

So the BitDepth{Setting,Params} is only an internal detail. It only means to
allow gemmlowp to use lower precision internally, but the input/output data
format is unaffected.

As far as the API contract goes, the only thing that the
BitDepth{Setting,Params} does is to relax the accuracy requirement. With
standard 8bit/8bit computation, gemmlowp is required to return the exact result
as specified in [low-precision.md](low-precision.md). With lower bit depths,
gemmlowp is no longer required to return an exact result.

## Implementation

Here we refer to the 3 stages of computation as described in
[design.md](design.md), namely: packing, computation kernel, unpacking.

The general idea is that at the packing stage, we requantize input (Lhs/Rhs)
data to less-than-8-bit depths by scaling them, thus shrinking the range of the
packed matrix entries; for instance, if the Rhs bit depth is to be 5 bits then
packed Rhs matrix entries will be in the range [0 ... 31]. This then allows the
GEMM kernel to use narrower accumulators without risking overflow, thus
achieving higher arithmetic throughput. Finally, at the unpacking stage, it only
remains to scale the result values to compensate for the scalings applied
earlier.

Let us go into more detail for each of those stages:

### Packing stage

The packing stage is where most of the work specific to the BitDepthParams takes
place.

Here, we have to scale input matrix values from their original range of [0 ...
255] to the range specified by the BitDepthParams, which is [0 ... (2^N)-1]
where N is the number of bits for the matrix at hand (Lhs or Rhs). For example,
for a bit depth of 5 bits, we need to scale down to [0 ... 31].

This scaling is what we call "requantization". The pedantic name matches the
fact that this is actually quite nontrivial to do correctly i.e. in such a way
that the result accuracy will be good enough for real-world applications. See
the section below on requantization details.

Concretely, this work happens in PackingRegisterBlock::Pack(), which calls
Requantize(). This is in internal/pack.h. This code can be overridden for
specific architectures, see internal/pack_neon.h.

This requantization work is costly and makes packing slower. This means that, at
least in our approach, less-than-8-bit computation is only interesting for
large-enough, square-enough GEMMs where packing is only a small fraction of the
overall cost. In cases where packing overhead is more prevalent (highly
rectangular cases), less-than-8-bit is probably a waste of time as long as we
treat it as an internal computation detail. What might help there, might be if
we shrunk the input/output data format to lower memory bandwidth usage.

### Computation kernel stage

In principle, the computation kernel stage simply doesn't have to care about the
bit depth at all. In fact, on architectures where we do not have specific
optimized kernels for less-than-8-bit cases, we simply use our standard kernel
there, and that's correct!

However, while the kernel doesn't have to know about the fact that the operands
are on less than 8 bits, it can use that information to make special
optimizations that would be incorrect in the general 8-bit case and become
correct here thanks to the more restricted range of inputs. That's the whole
point of this less-than-8-bit computation idea.

With Lhs entries guaranteed to be smaller than 2^N, and Rhs entries guaranteed
to be smaller than 2^M, each product is thus guaranteed to be smaller than
2^(M+N). Thus, one may accumulate 2^(16-(M+N)) such products and still be
guaranteed that such an accumulator will be smaller than 2^16, and thus can be
stored as a uint16 without risking overflow.

For example, in the L7R5 case, the Lhs enties are on 7 bits (N=7) and the Rhs
entries are on 5 bits (M=5), so each product fits in 12 bits and one can thus
accumulate 16 ( = 2^(16-12)) such products into uint16 accumulators with no risk
of overflow.

This means that a computation kernel may use uint16 accumulators for several
loop iterations (16 in the above example), provided that it is allowed to assume
that inputs are in such restricted range.

After this fixed number of loop iterations, the kernel must accumulate the local
uint16 accumulators back into global uint32 accumulators.

On SIMD architectures with suitable uint16 arithmetic, this in principle allows
to multiply arithmetic throughput by up to 2x, since twice more accumulators now
fit in each SIMD vector register. This is partially offset by the cost of
accumulating back into global uint32 accumulators every several loop iterations,
but our experience on ARM NEON has been that we still get quite close to a 2x
speedup. See internal/kernel_neon.h, specifically
NEON32Kernel12x4Depth2Assuming12BitProducts.

### Unpacking stage

At the unpacking stage, it only remains to scale the result values to compensate
for the scaling of the inputs. This is easier because now we are expanding the
range instead of shrinking it, so we don't need to worry about ways to minimize
a loss of accuracy. We simply need to multiply result values by a constant
fraction, rounding to nearest.

Since the inputs were scaled by factors of (2^lhs_bits - 1)/255 and
(2^rhs_bits - 1)/255 respectively, the scaling of the outputs needs to be by the
following factor:

                 255 * 255
    -----------------------------------
    (2^lhs_bits - 1) * (2^rhs_bits - 1)

This is done by a MultiplyByConstantFraction function, see internal/unpack.h

## Requantization details

Here we go into more detail on the Requantize() function used at the packing
stage to requantize input matrix data. See this function in internal/pack.h.

It depends on the bit depth and on a rounding mode, and requantizes an input
value in [0 ... 255] to the range [0 ... (2^N)-1] specified by the bit depth N.

### Naive, bad rounding, that's plainly biased

Naive and inaccurate ways to achieve this requantization include: 1. By shifting
bits rights by (8-N) bits; 2. By multiplying by ((2^N) - 1) and dividing by 255.

Both of those are biased in some way: 1. has the wrong "derivative", since it
approximates (((2^N) - 1) / 255) by ((2^N) / 256) ; 2. has bias since it
effectively implements rounding towards 0.

In practice, both of the above requantization functions give results that are
too inaccurate in practice for the neural network that we tried (GoogLeNet).

### Round-to-nearest rounding: unbiased in principle but not in practice

The simplest fix is to avoid the bias in 2. by rounding-to-nearest instead of
rounding towards 0. This can be achieved by doing

dst = (src * maxval + rounding_offset) / 255;

Where maxval = ((2^N) - 1) is the highest requantized value, and the
rounding_offset can be set to

rounding_offset = 127

to achieve rounding-to-nearest (while the above rounding towards 0 corresponded
to rounding_offset = 0).

In principle, rounding-to-nearest is unbiased and optimal in various ways.

In practice though, our input data is not random real numbers, but
already-quantized 8-bit values. That means that even in the best case, there
would be at most 255 different possible input values; in practice, we generally
see the input values distributed non-uniformly in that range, so that a majority
of input values tend to be in a much smaller range. See test/test_data.cc.

Having a large part of the input values in a very small finite set, means that
the corresponding rounding errors are also in a very small finite set, which can
be small enough that the mean of these rounding errors is significantly
different from 0. That rounding-to-nearest is "unbiased" only means that over a
sufficiently large set of input values, the bias would become arbitrarily close
to 0; here, the set of input values is effectively small enough that the
resulting bias is significant.

This leads to biasing the matrix product entries, resulting in an error that
grows linearly with the depth dimension of the GEMM.

### Probabilistic rounding: unbiased even on small finite input distributions

To address that, we can instead use probabilistic rounding. The idea is that for
instance if we have to round the value 3.8 to the nearest integer, we can round
it to 3 with 20% probability and to 4 with probability 80%. If that value 3.8
occurs many times, the mean requantized value will thus tend to 3.8.

This amounts to keeping the above requantization formula,

dst = (src * maxval + rounding_offset) / 255;

but now the rounding_offset is a random value in [0 .. 254].

This guarantees zero bias no matter how small the distribution of input values
is.

On the other hand, the variance of the error term here is higher than with
rounding-to-nearest --- one can check that it is 2x higher.

So the error term coming from the Central Limit Theorem, which grows with the
square root of the accumulator depth i.e. the GEMM depth, will be 2x higher
here.

Still, for large enough GEMM depth, that is better than rounding-to-nearest
which has an error term growing linearly with the GEMM depth.

### Switching between rounding-to-nearest and probabilistic rounding

Thus, for fixed input values and bit depths, we expect that probabilistic
rounding will give more accurate results for large enough GEMM depths, while
rounding-to-nearest will be more accurate for smaller GEMM depths.

That is why use switch between these rounding modes based on GEMM depth, see
ChooseRoundingMode in internal/bit_depth_util.h.

It is based on a constant, kProbabilisticRoundingThreshold, defined in
internal/common.h and empirically determined. See the comment there. It would be
nice to better understand the statistics here and come up with better heuristics
for this switching.

### Choice of pseudorandom number generator

We provide two PRNGs. The first is an 8-bit Xorshift. It is fast, naturally
produces values ranging over an interval of width 255, which is what we need
here (as opposed to an interval of width 256), and turns out, from empirical
tests, to produce better results than a linear congruential generator (LCG).
That's unfortunate, as a 8-bit LCG performs better (we confirmed that on various
ARM devices) but we need as perfect un-biased-ness as we can get.

The second is an "add-mod" sequence generator, which generates non-random values
in the sequence x = (x + 97) % 255. This generates a low-discrepancy sequence
that minimizes the "clumpiness" of the random offsets (Thus, for example,
quantizing a 3x3 matrix will have a maximum additive error of about 200 from the
random offsets). While not random, this sequence performs well empirically for
many quantizations. (For information about why 97 is a good value, see
https://en.wikipedia.org/wiki/Low-discrepancy_sequence#Additive_recurrence and
http://mollwollfumble.blogspot.com/2011/03/subrandom-numbers.html 97/255 = 0.38;
0.382 is the best choice. For discrete numbers, the choice must be relatively
prime to the modulus. 97 is prime, so it is safely relatively prime to 255. 107
is another near-optimal choice.

The low-discrepancy sequence generator is the default.

More details and results are given in a comment on the default PRNG in
internal/pack.h. Interested users can change the PRNG used by setting
DefaultRoundingGenerator in bit_depth_util.h.
