# Design: Selecting Tensors to Compress

## Background

Some TFLM operators can read tensors which have been reduced in size by
*Lookup-Table (LUT) Compression*. To create such a compressed tensor, its
elements' values, which are ordinarily of any value encodable in the data type
of the tensor, are *binned*[^1] to a smaller set of carefully chosen values.
Those values are still encoded in the data type of the tensor. The tensor is
then *compressed* by replacing each element with an index into a lookup table
containing the smaller set of values. The indices are encoded in a smaller data
type than that of the original tensor, therefore the resulting tensor is
smaller---i.e., compressed.

A model in which some tensors have been compressed, and with metadata added to
describe the compression, is a *compressed model*. A compressed model is no
longer readable by software which expects a standard models in TFLite flatbuffer
format.

It is useful to separate the creation of compressed models into the *binning*
stage and the *compression* stage. During the binning stage, the element values
are transformed in-place and retain their original data types, albeit a
restricted set of values of the data type. The output of this stage, a *binned
model*, remains in a standard format and therefore is compatible with existing
software. This is as advantage while developing a model---choosing which tensors
to compress, choosing the restricted set of values for the bins of each
compressed tensor, and testing the resulting model.

A separate *compression* stage rewrites the binned model: creates the lookup
tables, rewrites the binned tensors as indices into the lookup tables, and
writes other metadata to the model describing the compression. The result is a
compressed model, ready for use by the TFLM interpreter and other software
that can decompress compressed tensors as necessary.

In TFLM, only certain operators are capable of decompressing compressed tensors.
It is invalid to compress an input tensor of an operator which is not capable of
decompression.

## Problem Statement

The compression stage must discover or be told which tensors to compress.

## Proposed Design

The compression stage will:

1. By default, automatically compress any tensor that can be compressed.

1. Allow tensors to be excluded from consideration by a command-line option.

1. Disable automatic discovery and take an explicit list of tensors to compress
   by command-line option.

The compression stage will automatically try to compress any tensor that is used
only as an input to operators which are capable of decompression. If the total
size of the lookup table plus the size of the tensor elements, rewritten as
indices into the lookup table, is smaller than the size of the original tensor,
the tensor will be compressed.

The binned values of a tensor are discovered heuristically, by gathering the
set of unique values into a lookup table. 

The data type of the indices---i.e., the bit width of the unsigned integers used
for the indices---is determined by the number of unique values discovered in the
tensor as written by the binning stage. This width is constrained by the
implementation of the operators. If the set of unique values in a tensor cannot
be indexed by an integer with a bit width implemented by all the operators to
which the tensor is an input, the tensor will not be compressed.

The compression stage will output, in addition to the compressed model, a
description of which tensors have been compressed and with what bit widths.

## Alternative Designs

1. The list of tensors to compress could be communicated via metadata added to
   the model by the binning stage, rather than via command-line options.

1. The list of tensors to compress could be communicated as string matches to
   tensor names, rather than by index.

1. The binned values and or bit-widths could be communicated instead of
   automatically discovered.

1. The communication of options could be done via a configuration file rather
   than via command-line option.

---
[^1]: The word *quantization* is being avoided and a new word, *binning*, is
    used, because quantization typically refers to quantization of floating
    point values to the nearest point on a uniform grid of discrete values
    indexed by an integer data type; however, the general idea is similar.
