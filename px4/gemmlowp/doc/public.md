# Gemmlowp's public entry points

gemmlowp's public interface is defined in
[public/gemmlowp.h](../public/gemmlowp.h).

## GemmWithOutputPipeline

The primary public entry point is: `GemmWithOutputPipeline`.

A usage example is given in
[doc/quantization_example.cc](quantization_example.cc).

The high-level overview of how this specifies a low-precision matrix
multiplication is explained in [low-precision.md](low-precision.md). The
rationale for a specific quantization paradigm is given in
[quantization.md](quantization.md). That specific quantization paradigm is
implemented at two different stages of the computation: as pre-processing ont
the operands and as post-processing on the result:

*   Pre-processing on the LHS, RHS operands, in the form of adding constant
    `lhs_offset`, `rhs_offset` to them, is explained in
    [low-precision.md](low-precision.md).

*   Post-processing on the result, in the form of a flexible "output pipeline",
    is explained in [output.md](output.md).

More details on this below as we discuss specific function parameters.

The prototype is:

```
template <typename InputScalar, typename OutputScalar, typename BitDepthParams,
          MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder,
          typename OutputPipelineType, typename GemmContextType>
void GemmWithOutputPipeline(GemmContextType* context,
                            const MatrixMap<const InputScalar, LhsOrder>& lhs,
                            const MatrixMap<const InputScalar, RhsOrder>& rhs,
                            MatrixMap<OutputScalar, ResultOrder>* result,
                            int lhs_offset, int rhs_offset,
                            const OutputPipelineType& output_pipeline);
```

A typical call looks like (from the [usage example](quantization_example.cc)):

```
gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::uint8_t,
                                 gemmlowp::DefaultL8R8BitDepthParams>(
    &gemm_context, uint8_lhs_matrix, uint8_rhs_matrix,
    &uint8_result_matrix, lhs_offset, rhs_offset, output_pipeline);
```

### Template parameters

Typically only the 3 first template parameters need to be specified, the rest
being automatically deduced from function parameters:

*   `InputScalar`: The scalar type of the LHS and RHS operands. At the moment,
    this must be `std::uint8_t`.
*   `OutputScalar`: The scalar type of the LHS and RHS operands. At the moment,
    this must be `std::uint8_t`.
*   `BitDepthParams`: Defines the bit format of the input and output matrices
    and the required accuracy of the computation. At the moment, the only
    non-deprecated valid value is `gemmlowp::DefaultL8R8BitDepthParams`. See
    [less-than-8-bit.md](less-than-8-bit.md) for other values and the general
    idea of this, and how it may become more useful in the future.

The other template parameters, which typically do not need to be specified, are:

*   `LhsOrder`, `RhsOrder`, `ResultOrder`: the storage orders (row-major or
    column-major) of the LHS, RHS, result matrices. See
    [public/map.h](../public/map.h). See the below performance note: we
    recommend using respectively RowMajor, ColMajor, ColMajor for optimal
    performance.
*   `OutputPipelineType`: the actual `std::tuple` type of the output pipeline.
    See below explanation of the `output_pipeline` parameter, and
    [output.md](output.md).
*   `GemmContextType`: the type of the `context` parameter. At the moment, this
    must be `gemmlowp::GemmContext`.

### Function parameters

The function parameters taken by `GemmWithOutputPipeline` are:

*   `context`: The `gemmlowp::GemmContext` object holding state and resources to
    be used for this gemmlowp call.
*   `lhs`, `rhs`: The LHS and RHS operand matrices. Note that these are
    `MatrixMap` objects, mapping external buffers as matrices, not owning data.
    See [public/map.h](../public/map.h).
*   `result`: pointer to the destination `MatrixMap` object, which must be
    already constructed, wrapping the external destination buffer with the
    wanted destination matrix shape and storage layout. No memory allocation
    will be performed by gemmlowp for the destination buffer. See
    [public/map.h](../public/map.h).
*   `lhs_offset`, `rhs_offset` are constants added to each matrix entry in the
    LHS, RHS matrices respectively, as explained in
    [low-precision.md](low-precision.md). This is only the part of the
    quantization paradigm explained in [quantization.md](quantization.md) that
    needs to be implemented as operations on the operands; everything else is
    operations on the result, see `output_pipeline`.
*   `output_pipeline` is a `std::tuple` of output stages (see
    [public/output_stages.h](../public/output_stages.h)), specifying the output
    pipeline (see [output.md](output.md)). This is the part of the quantization
    paradigm explained in [quantization.md](quantization.md) that needs to be
    implemented as operations on the result matrix.

### Performance note on storage orders.

gemmlowp supports arbitrary combinations of storage orders for the LHS, RHS and
result matrices. However, not all are equally optimized for.

Because gemmlowp is primarily aimed at neural network inference workloads,
optimization focus is on this particular combination of storage orders:

*   `LhsOrder=RowMajor`
*   `RhsOrder=ColMajor`
*   `ResultOrder=ColMajor`

The rationale is that the LHS is typically the constant weights of a neural
network layer (e.g. the weights of a Convolutional layer implemented as a matrix
multiplication), while the RHS and result are neural network activations,
respectively the input and output activations of the layer.

Because the RHS and result are activations, we want them to share the same
storage order -- so that one layer's output activations can be readily used as
the next layer's input activations. Thus, we focus on `RhsOrder=ResultOrder`.

We also know from general considerations on matrix multiplication that it is
slightly more efficient to have the direction of accumulation (the "depth"
dimension) be the direction of contiguous storage in memory. That means that it
is always going to be slightly easier and more efficient to have
`LhsOrder=RowMajor` and `RhsOrder=ColMajor`.

Putting this together, we arrive at gemmlowp's focus on the above-described
combination of storage orders.

Using other storage orders will typically mean taking less efficient paths in
the packing and unpacking stages, see [packing.md](packing.md). The compute
kernel stage ([kernel.md](kernel.md)) is unaffected.

## GemmWithOutputPipelinePC

This is a variant where `lhs_offset` and `rhs_offset` may be vectors instead of
scalar. They are then broadcasted against LHS, RHS respectively.

This is useful for some flavors of neural network inference with "per-channel
quantization", whence the PC suffix. This has been useful in some settings where
a neural network trained in float arithmetic was subsequently quantized. On the
other hand, retraining neural networks for quantized inference tends to remove
the need for per-channel quantization. For that reason, the long-term usefulness
of this entry point is in question.

## Gemm

This is gemmlowp's original, now legacy and deprecated, entry point. See the
section of [low-precision.md](low-precision.md) on the legacy quantization
paradigm. Avoid in new code.

## The eight_bit_int_gemm directory

As explained in the top-level [README.md](../README.md#public-interfaces), this
is entirely deprecated.
