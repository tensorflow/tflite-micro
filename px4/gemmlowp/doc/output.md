# Output pipelines in gemmlowp

In gemmlowp, the "output pipeline" is the process that takes a final `int32`
accumulator value (the output of the compute/kernel stage), and processes it to
obtain the final value (typically a `uint8` value) and write it to the
destination matrix.

Gemmlowp has some genericity in what arithmetic transformations take place in
the output pipeline, so as to allow different users to implement different
quantization paradigms. See [low-precision.md](low-precision.md) and
[quantization.md](quantization.md).

Besides implementing a quantization paradigm, the other thing that output
pipelines is good for, is implementing fused operations where a matrix
multiplication feeds into other operations applied to its result, without
additional array traversals. For instance, when implementing neural network
inference, one might have a Convolutional layer with a bias-addition and an
activation. One then wants to feed the result of the matrix multiplication
implementing the Convolutional operator itself, directly into the bias-addition
and activation function. gemmlowp's output pipelines allow implementing that:
the bias-addition and activation function are just additional stages in the
output pipeline.

## Usage

The gemmlowp entry point allowing to use an arbitrary output pipeline is
`GemmWithOutputPipeline` in [public/gemmlowp.h](../public/gemmlowp.h).

The output pipeline is specified as a `std::tuple` of "output stages", each of
which defining an elementary arithmetic transformation.

All available output stages are defined in
[public/output_stages.h](../public/output_stages.h).

## Example usage

The best part to see examples of using various output pipelines is in the unit
test,

```
test/test.cc
```

specifically in this function:

```
TestOutputStages
```

Separately, a self-contained example showing how to use gemmlowp to compute a
quantized matrix multiplication with a sounds quantization paradigm, is here:

[doc/quantization_example.cc](quantization_example.cc)
