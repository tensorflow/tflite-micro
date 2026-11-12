# TFLM Compression Support

TFLM supports fixed width compression of const-tensors using lookup tables and
the `DECODE` custom operator.
Const-tensors are typically those containing trained weights or biases, but can
be any tensor where the values are fixed within the model and unchanging.

For a complete end-to-end tutorial demonstrating compression with weight clustering,
see the [MNIST Compression Tutorial](../tensorflow/lite/micro/compression/tutorial/mnist_compression_tutorial.ipynb).

Const-tensors are compressed to fixed width bitstrings, and lookup tables are
added to the model schema along with a `DECODE` operator node that decompresses
tensors during inference.

When executing a `DECODE` operator node, each set of fixed width bits in the
compressed tensor bitstring is used as an index into the tensor lookup table.
The results of the lookup table operations are placed into the output tensor
buffer (in either the interpreter arena or an alternate decompression memory
region) for consumption by downstream operators.

Decompression results in increased latency during inference.
There will also be an increase in the size of non-persistent arena memory (unless
alternate decompression memory is configured), due to the temporary buffers needed
to hold the decompressed tensor data.

# Supported Tensor Types

* FLOAT32, INT8, INT16, INT32, INT64, BOOL

# Supported Platforms

* Portable reference implementation (all platforms)
* XTENSA optimized implementations
  * P6_VISION, HIFI3, HIFI4, HIFI5

# Compressed Tensor Representation

## Tensor Bitstrings

Each compressed tensor has its buffer data replaced by a packed bitstring.  The
bitstring consists of fixed bit width groups (indices), each group representing an offset
into the `value table`.  The bitstring is in big-endian byte order with the most
significant bit first.  A bitstring is padded on the end, to the next byte
boundary, with zero bits.

Example (bit width 3):
```
1110000110100000
--|--|--|--|---|
  7  0  3  2   padding
```
This bitstring represents the indices 7, 0, 3, 2 as offsets into the `value table`.
Each offset is in the same units as the original (uncompressed) tensor.  So if
the tensor is INT8, each offset represents a byte in the `value table`. If the
tensor was FLOAT32, each offset would represent four bytes.

The indices in the bitstring are in the same order as the tensor's original data.
Compression never reorders the tensor data, simplifying the decompression phase.

## Value Tables

A `value table` contains the unique data values from an original (uncompressed)
tensor.  For each compressed tensor, an additional buffer is added to the model,
and the `value table` resides as a contiguous sequence of data values within
that buffer.  Each element in the `value table` is unique, and is of the same type
(INT16, FLOAT32, etc.) as the uncompressed tensor.  The order of values within
the `value table` does not have to match the order in which they appeared in
the uncompressed tensor data.

Example (tensor type is INT16, value table size is 12 bytes):
```
tensor data: [2, 4, 4, 10, 1, 7, 99, 10, 2, 4]
value table: [99, 2, 10, 4, 1, 7]
```
A suitable tensor bitstring (bit width 3) for the example would be:
```
bitstring: 00101101101010010100001000101100
             |  |  |  |  |  |  |  |  |  | |
index:       1  3  3  2  4  5  0  2  1  3 padding
value:       2  4  4 10  1  7 99 10  2  4
```

### Per-channel Quantized Tensor Value Tables

For per-channel quantized tensors, a `value table` is present for each channel.
All of the `value tables` are concatenated together into a single contiguous
set of values. The number of elements in each `value table` is always identical,
with zero value padding added to the end of a `value table` as necessary.

Using the previous example tensor (above) with 2 channels:
```
tensor data: [2, 4, 4, 10, 1, 7, 99, 10, 2, 4]
channel:      |______0_____|  |______1______|
              |            |  |             |
value table: [1, 10, 2, 4, 0, 99, 10, 2, 7, 4]
                           |
                           |__padding
```
A suitable tensor bitstring (bit width 3) for the example would be:
```
bitstring: 01001101100100001100000101010000
             |  |  |  |  |  |  |  |  |  | |
index:       2  3  3  1  0  3  0  1  2  4 padding
value:       2  4  4 10  1  7 99 10  2  4
channel:     0  0  0  0  0  1  1  1  1  1
```

Note that in the above example, compressed tensor indices are specific to a `value table` channel.

Also note that channel 0 (zero) in the `value table` is padded with a single
zero value at the end.

# The MicroInterpreter and Tensor Decompression

To run a compressed model, register the `DECODE` custom operator with your
`MicroMutableOpResolver`:

```cpp
op_resolver.AddCustom(tflite::kCustom_DECODE, tflite::Register_DECODE());
```

### Alternate Decompression Memory

Alternate decompression memory regions allow the use of specialized memory
available to the processor, to be used as the target of a tensor decompression
operation.  Such memory is typically mapped by the application through a linker
script.  The application would then use a C++ attribute of the form:
```
__attribute__((section(".your-specialized-memory")))
```
to link one or more application symbols to the specialized memory region.

Only a single API is required to use alternate decompression memory regions in
an application:
* `MicroInterpreter::SetDecompressionMemory` ([micro_interpreter.h](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_interpreter.h)):
Specify the address and size of each alternate decompression
memory region.  This method must be called before the application calls
`MicroInterpreter::AllocateTensors`.  The lifetime of the method parameter must
equal the lifetime of the `MicroInterpreter` instance.  The memory regions
specified by the method parameter must not overlap, and each region is considered
to be non-contiguous with all other regions.

Specifying alternate decompression memory will cause the `DECODE` operator
(via `MicroContext::AllocateDecompressionMemory`) to attempt to allocate memory
for the decompression destination buffer from available memory in one of the
alternate memory regions.  If no alternate memory region of sufficient size is
available, the output tensor buffer in the `MicroInterpreter` arena is used.

# How to Compress a Model

Compression works best when the targeted tensors in the model have been binned.
Binning of the model tensors will result in a change in model accuracy, but will
also allow for better control of the compression ratio.  For example, by binning
a tensor to just four values among the tensor elements, a fixed-width of two bits
can be used for each element.  This would result in nearly a four-fold decrease
in the size of an INT8 tensor.

Tensors to compress are specified with a `YAML` file.  For example, if tensors
5, 10, 11, 22 of subgraph 0 of the model are to be compressed, the contents of
the file would be as follows:
```
tensors:

  - subgraph: 0
    tensor: 5
    compression:
      - lut:
          index_bitwidth: 4
          per_channel:
            axis: 0

  - subgraph: 0
    tensor: 10
    compression:
      - lut:
          index_bitwidth: 4
          per_channel:
            axis: 0

  - subgraph: 0
    tensor: 11
    compression:
      - lut:
          index_bitwidth: 2
          per_tensor:

  - subgraph: 0
    tensor: 22
    compression:
      - lut:
          index_bitwidth: 2
          per_tensor:
```
Note that each tensor can have a different bit width (1 through 7 bits).

Each `lut` entry states its compression mode, exactly one of `per_channel`
or `per_tensor`. `per_channel` builds one value table per channel, along
the given axis of the tensor's shape. A bare `per_tensor:` builds one
value table for the whole tensor.

Once the `YAML` specification is ready, compress the model using the following:
```
bazel run -s tensorflow/lite/micro/compression:compress -- --input=binned.tflite --output=compressed.tflite --spec=spec.yaml
```

# The Generic Benchmark Application

The Generic Benchmark Application can be used to see the size of the model, the
amount of arena memory used, and the size of the interpreter data structures
including those involved with tensor compression. The benchmark also reports
total inference time, as well as time taken for tensor decompression.

For additional information on the Generic Benchmark Application, please refer to
this [document](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools/benchmarking/README.md).
