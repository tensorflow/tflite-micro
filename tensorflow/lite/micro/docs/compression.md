# TFLM Compression Support

TFLM supports fixed width compression of const-tensors using lookup tables.
Const-tensors are typically those containing trained weights or biases, but can
be any tensor where the values are fixed within the model and unchanging.

Const-tensors are compressed to fixed width bitstrings, and lookup tables are
added to the model schema for each tensor.

When accessing a compressed tensor, each kernel invokes a common decompression
method.  Each set of fixed width bits in the tensor bitstring are used as
indices into the tensor lookup table.  The results of the lookup table operations
are placed into a scratch buffer representing the tensor decompressed data.

Decompression results in increased latency during inference.
There will also be an increase in the size of non-persistent arena memory, due to
the use of scratch buffers to temporarily hold the decompressed data.

# Supported Tensor Types

* FLOAT32, INT8, INT16, INT32, INT64, BOOL

# Supported Kernels

* FULLY_CONNECTED
* CONV_2D
* DEPTHWISE_CONV
* TRANSPOSE_CONV
* CONCATENATION
* ASSIGN_VARIABLE

Per-channel quantized tensor support is available for:
* CONV_2D
* DEPTHWISE_CONV
* TRANSPOSE_CONV
* FULLY_CONNECTED

# Supported Platforms

* X86
* XTENSA
  * P6_VISION, HIFI_MINI, HIFI3, HIFI4, HIFI5

# Model and Metadata Schema for Compression

Models that use compression will have a string key in their `Metadata` vector
corresponding to `COMPRESSION_METADATA`.  The buffer indexed by such a `Metadata`
entry will contain the compression schema.  The complete compression schema can
be found [here](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/compression/metadata.fbs).

For each tensor which is compressed, the following schema element is created:
```
table LutTensor {
    // Look-Up-Table Tensor: a tensor representation where elements are
    // compressed into indices into a table of values. The indices are unsigned
    // integers, index_bitwidth-wide, in big-endian bit order, packed into the
    // buffer identified by the corresponding tflite.Tensor's buffer field. The
    // values are located in a newly-created buffer, encoded according to the
    // tflite.Tensor.type. Tensors with multiple channels have distinct values
    // tables for each channel, concatenated one after another in the buffer.
    // An element's LUT index must be looked up in the value table for its
    // channel.

    tensor:int;            // index of the corresponding tflite.Tensor
    value_buffer:uint;     // index of the buffer containing LUT values
    index_bitwidth:uint8;  // bit-width of LUT indexes
}
```

* `tensor`: the index of the tensor in the current subgraph.  This tensor will
have had its buffer data replaced with a packed bitstring (see below),
representing fixed width indices into the `value table`.
* `value_buffer`: the index of a buffer added to the model.  This buffer contains
the `value table` (see below) for the tensor, which is used to decompress the
tensor.  The elements of the `value table` are of the same type (INT8, INT16, etc.)
as the original (uncompressed) tensor.
* `index_bitwidth`: the fixed width of each bit group (index) that represents an offset
into the `value table`.  For per-channel quantized tensors, the index is an
offset into the `value table` for a specific channel.

## Tensor Bitstrings

Each compressed tensor has its buffer data replaced by a packed bitstring.  The
bitstring consists of fixed bit width groups (indices), each group representing an offset
into the `value table`.  The bitstring is in big-endian byte order with the most
significant bit first.  A bitstring is padded on the end, to the next byte
boundry, with zero bits.

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

While the compressed tensor data buffer will shrink in size, the tensor shape
(dimensions) will remain the same as the uncompressed tensor.

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

The model schema `Metadata` is first searched for the `COMPRESSION_METADATA` key.
If found, the associated buffer is decoded using the [compression schema](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/compression/metadata.fbs).  For each `LutTensor` in the compression schema,
a `LookupTableData` ([compression.h](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/compression.h))
structure is instantiated.

```cpp
struct LookupTableData {
  static constexpr size_t kMaxBitWidth = 7;
  static constexpr size_t kMaxValueTableChannelStride = 128;

  const void* value_table;             // Pointer into FlatBuffer Values.
  uint8_t value_table_channel_stride;  // elements per channel
  uint8_t compressed_bit_width : 3;    // 1 to 7 bits
  bool is_per_channel_quantized : 1;   // tensor is per-channel quantized
  bool use_alternate_axis : 1;         // shape default channel:
                                       // 0 = first, 1 = last
  uint8_t reserved : 3;
};
```

* `value_table`: Pointer to the buffer memory containing the `value table`.
Determined from the `LutTensor.value_buffer` and converted to a model schema
buffer vector.
* `value_table_channel_stride`: The number of elements (not bytes) between
`value table` channels.  Only valid for per-channel quantized tensors.
* `compressed_bit_width`: Number of bits for each `value table` index.
Determined from `LutTensor.index_bitwidth`.
* `is_per_channel_quantized`: Will be `true` for per-channel quantized
tensors.  Determined by inspecting the tensor quantization scale vector size in
the model schema.
If the vector size is greater than 1 (one) then the tensor is assumed to be
per-channel quantized.
Default value is `false`.
* `use_alternate_axis`: Arrangement of tensor data vs. channel number.
See the quantized dimension section below for additional explanation.
Only valid for per-channel quantized tensors.  Default value is `false`.

## Quantized Dimension

Each per-channel quantized tensor will have as part of its model schema quantization
information, a `quantized_dimension` field.  This field specifies which dimension
of the tensor shape along which the scale and zero-point are to be applied. This
dimension within the shape is sometimes referred to as the `quantization axis`.

The importance of the `quantization axis` is in how the
tensor data is interpreted with respect to channel number.
The tensor decompression methods use `LookupTableData.use_alternate_axis` to
determine the correct `value table` channel for each tensor element.  When the
`quantized_dimension` field is 0 (zero) then `use_alternate_axis` is `false`.
If the `quantized_dimension` field is set to 3 (three) (ex. DEPTHWISE_CONV), then
`use_alternate_axis` will be `true`.

For a tensor with shape [4, 2, 2, 1] and `use_alternate_axis` equal to `false`,
the tensor data is assumed to be arranged as follows:
```
element number:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
channel number:  0  0  0  0  1  1  1  1  2  2  2  2  3  3  3  3
```

For a tensor with shape [1, 2, 2, 4] and `use_alternate_axis` equal to `true`,
the tensor data is assumed to be arranged as follows:
```
element number:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
channel number:  0  1  2  3  0  1  2  3  0  1  2  3  0  1  2  3
```

## Decompressing a Tensor

Any kernel can have decompression support easily added.
Tensor data is decompressed into the designated memory buffer, and is available
for the lifetime of the memory buffer.

Only the following methods are required to implement decompression within kernel code:

* `MicroContext::AllocateDecompressionScratchBuffer` ([micro_context.h](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_context.h)):
Allocates a scratch memory buffer within the `MicroInterpreter` to hold the
decompressed tensor data.  The returned scratch memory handle must be retained
(typically through kernel `OpData`) for use during the kernel inference operation.
* `MicroContext::GetTensorCompressionData` ([micro_context.h](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/micro_context.h)):
Retrieves compressed tensor information (see [compression.h](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/compression.h)).
* `tflite::micro::GetTensorData` ([kernel_util.h](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/kernel_util.h)):
The four parameter version of this method will automatically decompress the
tensor data into the supplied scratch memory buffer.  The lifetime of a scratch
buffer is the same as the lifetime of the current kernel operator being processed.
Each call to the four parameter version of this method will always result in a
decompression operation being performed, if the tensor supplied is compressed.

Please see the [TRANSPOSE_CONV](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/kernels/transpose_conv.cc)
reference kernel code for an example of how to implement tensor decompression
within a kernel.

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

Specifying alternate decompression memory will cause `MicroContext::AllocateDecompressionScratchBuffer`
and `tflite::micro::GetTensorData` (the four parameter version)
to automatically attempt to allocate memory for the decompression destination
buffer from available memory in one of the alternate memory regions.  If no
alternate memory region of sufficient size is available, a scratch buffer will
be allocated within the `MicroInterpreter` arena.

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

  - subgraph: 0
    tensor: 10
    compression:
      - lut:
          index_bitwidth: 4

  - subgraph: 0
    tensor: 11
    compression:
      - lut:
          index_bitwidth: 2

  - subgraph: 0
    tensor: 22
    compression:
      - lut:
          index_bitwidth: 2
```
Note that each tensor can have a different bit width (1 through 7 bits).

Once the `YAML` specification is ready, compress the model using the following:
```
bazel run -s tensorflow/lite/micro/compression:compress -- --input=binned.tflite --output=compressed.tflite --spec=spec.yaml
```

Then align the model:
```
bazel run -s tensorflow/lite/micro/tools:tflite_flatbuffer_align -- compressed.tflite compressed_and_aligned.tflite
```

# The Generic Benchmark Application

The Generic Benchmark Application can be used to see the size of the model, the
amount of arena memory used, and the size of the interpreter data structures
including those involved with tensor compression. The benchmark also reports
total inference time, as well as time taken for tensor decompression.

For additional information on the Generic Benchmark Application, please refer to
this [document](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools/benchmarking/README.md).
