# Building a quantization paradigm from first principles

**TLDR:** If you prefer example code over theory, look at
[doc/quantization_example.cc](quantization_example.cc).

## Overview

gemmlowp allows to perform calculations on matrices on uint8 values, but these
matrices are only useful insofar as they somehow approximate matrices of real
numbers. By a _quantization paradigm_ we mean a correspondence between matrices
of quantized 8bit values and matrices of real numbers. The choice of a
quantization paradigm affects the calculations that gemmlowp itself needs to
perform, specifically, it affects how one goes from internal 32bit accumulator
to final 8bit outputs.

The part of gemmlowp transforming internal internal 32bit accumulator to final
8bit outputs is the "output pipeline" described in [output.md](output.md).

gemmlowp's `GemmWithOutputPipeline` entry point allows specifying an arbitrary
output pipeline, allowing the user to implement their own preferred quantized
arithmetic paradigm.

In the present document, our purpose is to show how, reasoning from first
principles and some domain-specific knowledge of neural networks, we can arrive
naturally at some specific quantization paradigm, and how that can be
implemented using a specific output pipeline.

We also aim to show how that differs from the older, legacy quantization
paradigm implemented by gemmlowp's legacy interfaces and why the change to the
newer quantization paradigm described in this document was useful as far as some
applications of gemmlowp were concerned.

## Quantization as an affine map.

In order for arithmetic on real values to map directly to arithmetic on
quantized uint8 values, the mapping between real and quantized uint8 values must
be affine, which means it must be of the form

```
real_value = A * quantized_value + B             (1)
```

for some constants A, B, or equivalently, of the form

```
real_value = C * (quantized_value + D)           (2)
```

for some constants C, D. Indeed, anything else than such an affine map would
mean that the result of the quantized calculations do no longer readily provide
an approximation to the result of the real-numbers calculation.

## Domain-specific constraint: the real value 0 must be exactly representable.

Here a domain-specific constrain from neural networks appears: for some neural
network layers, it is very useful for optimized implementations that the
real-value 0 be exactly representable.

For instance, in a Convolutional or Pooling layer with padding, it is useful to
be able to implement the padding by zero-padding the input array, so that
optimized loops do not need to become more complex to avoid overrunning the
array bounds.

In order for such zero-padding to be feasible in a quantized implementation of
such layers, it is important that the real value '0' be exactly representable in
quantized form, i.e. that it correspond exactly to some quantized value, which
we call the _zero-point_.

Indeed, if '0' were not exactly representable, then we would have to use some
quantized value for padding, that does not exactly correspond to the real value
'0'. That would typically introduce inaccuracy in the result. In fact, using
always the same such value would be worse: it would introduce _bias_ in the
result.

## The final form of the quantization equation

Now let us phrase what this constraint &mdash; that the real value 0 be exactly
representable &mdash; means in either quantization equations, (1) and (2).

In equation (1), plugging `real_value = 0` and `quantized_value = zero_point`,
we get:

```
0 = A * zero_point + B
```

equivalently:

```
zero_point = -B / A
```

We are thus left with a rather awkward constraint: the real number `-B / A` must
somehow be guaranteed to be exactly integral, so that the special uint8 value
`zero_point` can be exactly equal to it. Quite awkward!

Now let us look at equation (2). Plugging `real_value = 0` and
`quantized_value = zero_point`, we get:

```
0 = C * (zero_point + D)
```

Conveniently, the constant `C` plays no role anymore, so this equation
simplifies to:

```
0 = zero_point + D
```

In other words, `D = -zero_point`. This suggests rewriting the quantization
equation (2) into the following form (3), which will be the final form that we
will consistently use:

```
real_value = scale * (quantized_value - zero_point)        (3)
```

To go from (2) to (3), we merely renamed `C` to `scale` and `D` to
`-zero_point`.

With this quantization equation (3), the condition that 0 be exactly
representable is vacuously satisfied: `zero_point` is by definition one of the
possible `quantized_value`'s, and equation (3) maps it to a `real_value` of
exactly 0.

Note that the final quantizaton equation (3) depends on two constants, one
integral, the other an arbitrary positive real number:

*   `zero_point` is integral, more specifically is one of the possible quantized
    values (i.e. typically is a uint8 value).
*   `scale` is a positive real number. Thus at this stage we have not yet shown
    how to eliminate all usage of floating-point arithmetic. That will come
    below.

## Quantizing a matrix multiplication

Now that we know &mdash; equation (3) &mdash; how real numbers are to correspond
to quantized values (typically uint8), we turn to applying this knowledge to
rewriting a multiplication of matrices of real numbers, by the equivalent
multiplication of matrices of quantized values.

Say that we have two matrices of real values `lhs_real_matrix`,
`rhs_real_matrix`. Each entry of their product is the sum (accumulation) of many
products of individual matrix entries, say `lhs_real_value * rhs_real_value`.

Now suppose that we have already quantized these two matrices according to the
above equation (3), with some already-known quantization parameters `lhs_scale`,
`rhs_scale`, `lhs_zero_point`, `rhs_zero_point`, so that their matrix entries
are quantized as

```
lhs_real_value[i] = lhs_scale * (lhs_quantized_value[i] - lhs_zero_point)
rhs_real_value[i] = rhs_scale * (rhs_quantized_value[i] - rhs_zero_point)
```

We then rewrite the matrix product accumulator accordingly:

```
result_real_value
  = Sum_over_i(lhs_real_value[i] * rhs_real_value[i])
  = Sum_over_i(
        lhs_scale * (lhs_quantized_value[i] - lhs_zero_point) *
        rhs_scale * (rhs_quantized_value[i] - rhs_zero_point)
    )
  = lhs_scale * rhs_scale * Sum_over_i(
        (lhs_quantized_value[i] - lhs_zero_point) *
        (rhs_quantized_value[i] - rhs_zero_point)
    )                                                      (4)
```

Now our goal is to represent this result itself as a quantized matrix, i.e.
still according to equation (3), for some pre-established quantization
parameters `result_scale` and `result_zero_point`, as

```
result_real_value = result_scale *
    (result_quantized_value - result_zero_point)
```

Here we need to keep in mind that our goal is to specify what the quantized
matrix multiplication should do, i.e. how to compute `result_quantized_value`.
The last equation above is equivalent to

```
result_quantized_value = result_zero_point +
    result_real_value / result_scale
```

Now we can use equation (4) above to plug into this the expression of
result_real_value in terms of the quantized operands, and we obtain:

```
result_quantized_value = result_zero_point +
    (lhs_scale * rhs_scale / result_scale) *
        Sum_over_i(
            (lhs_quantized_value[i] - lhs_zero_point) *
            (rhs_quantized_value[i] - rhs_zero_point)
        )                                                  (5)
```

Equation (5) is the conclusion of this general discussion of how to specify what
"quantized matrix multiplication" should actually compute, in order to be able
to replace real matrix multiplications.

## Implementation of quantized matrix multiplication

Having obtained the mathematical form (5) of quantized matrix multiplication, we
now turn to its actual implementation.

The inner-most part of (5),

```
int32_accumulator =
    Sum_over_i(
        (lhs_quantized_value[i] - lhs_zero_point) *
        (rhs_quantized_value[i] - rhs_zero_point)
)
```

is the "kernel" accumulation loop. It is where the bulk of the computational
cost goes. Luckily, it only involves integers: the quantized operands matrix
entries, and their `zero_point` quantization parameters. Typically, all of these
values are uint8. Typically, the above differences of uint8 values would be
represented as signed int16; their products as signed int32.

It is out of scope of the present doc to discuss how to avoid the overhead of
having to subtract these `zero_point` constants in this inner loop; refer to
[this section of
low-precision.md](low-precision.md#efficient-handling-of-offsets) for that. The
gist of it is that a mathematical trick allows us to take the handling of these
`zero_point` constants out of this accumulation loop, so that it simplifies to

```
int32_accumulator =
    Sum_over_i(
      lhs_quantized_value[i] *
      rhs_quantized_value[i]
    )                                                      (6)
```

Anyway, the result is a `int32_accumulator` that we now plug back into the rest
of (5):

```
result_quantized_value = result_zero_point +
    (lhs_scale * rhs_scale / result_scale) * int32_accumulator       (7)
```

The difficulty here is of course that `(lhs_scale * rhs_scale / result_scale)`
is a positive real number, not an integer in general. It is a constant, though.
So what we have to implement here is the (approximate) scaling of a int32 value
by some arbitrary positive constant multiplier.

Moreover, it is safe to assume that this positive constant multiplier is smaller
than one &mdash; each of the `scale` values here is typically smaller than one,
as we are typically mapping the `[0..255]` quantized uint8 value range to an
interval of real values that is much narrower than that, typically within
`[-10,10]` in most neural networks. For example, a neural network using Relu6
activation functions will typically have real activation values in the interval
[0,6].

So how do we implement the multiplication of a int32 value by a positive real
constant that is smaller than one? Typically, by multiplying by a fixed-point
constant multiplier in the normalized interval `[1/2,1)`, and right-shifting
the result to achieve the correct multiplier.

At this point we have obtained the int32 value of the product

```
(lhs_scale * rhs_scale / result_scale) * int32_accumulator
```

Looking at (7), it only remains to add to it the integral value
`result_zero_point`, and we are done.

## How this is implemented in gemmlowp

The different parts of gemmlowp implementing aspects of the above discussion
are:

*   The packing stage (see [packing.md](packing.md)) implements the special
    mathematical trick to handle `lhs_offset`, `rhs_offset` that we alluded to
    above, see [this section of
    low-precision.md](low-precision.md#efficient-handling-of-offsets) for
    details. Thanks to is, the rest of the calculation can proceed as if
    `lhs_offset`, `rhs_offset` were 0.

*   The compute/kernel stage (see [kernel.md](kernel.md)) performs the core
    accumulation loop producing the `int32_accumulator`, see equation (6) above.

*   The unpacking stage feeds into the output pipeline (see
    [output.md](output.md)), which implements the rest of the evaluation of the
    above equation (5), that we discussed in the previous section.

Now, the point of gemmlowp's flexible output-pipelines mechanism (see
[output.md](output.md)) is to support different quantization paradigms, so we
now have to specify which particular flavor of output pipeline corresponds to
the particular quantization paradigm that we detailed above in this document.

The specific output pipeline stage implementing the present quantization
paradigm, i.e. implementing the precise computation detailed in the previous
section (equation (5)), is
`OutputStageQuantizeDownInt32ByFixedPoint`.

Please refer to the comment explaining it in
[public/output_stages.h](../public/output_stages.h).

## How this differs from the older legacy gemmlowp quantization paradigm

The difference between the older legacy quantization paradigm described in
[low-precision.md](low-precision.md) and the newer one described in this
document boils down to the difference between the legacy output stage
implementing it, `OutputStageQuantizeDownInt32ToUint8Scale`, and the new output
stage implementing the new paradigm,
`OutputStageQuantizeDownInt32ByFixedPoint`.

Please refer to the comments in
[public/output_stages.h](../public/output_stages.h) for details about these two
output stages and how they differ.

Issues with the old output stage `OutputStageQuantizeDownInt32ToUint8Scale` are:

1.  The int32 accumulators (inputs to the output stage) undergo a plain int32
    multiplication with a int32 multiplier, which may overflow. By contrast, in
    the newer `OutputStageQuantizeDownInt32ByFixedPoint`, this
    integer multiplication becomes a fixed-point multiplication and cannot
    overflow.

    *   In practice, to limit the risk of overflow, this pushes users to choose
        smaller values for this integer multiplier, which means limited
        multiplicative accuracy, which may cause multiplicative bias depending
        on how it is used.

2.  Note how the order of multiplying by the multipler and adding the
    `result_offset` are swapped. This reflects a quantizatin equation of the
    form (1) above, as opposed to the form (2)/(3) that the new quantization
    paradigm uses. As a result, it is essentially impossible to guarantee that 0
    is an exactly-representable value, which as discussed above is an issue at
    least in some convolutional neural network applications.

## Example code illustrating the new quantization paradigm

Example code showing how to perfom a quantized matrix multiplication in the
quantization paradigm discussed here is in
[doc/quantization_example.cc](quantization_example.cc).
