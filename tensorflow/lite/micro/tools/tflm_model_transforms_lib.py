"""TFLM specific flatbuffer model transformations, to reduce model size.

go/tflm-flatbuffer-reduction
We take advantage of the TFLM infrastructure to remove information in the
flatbuffer which we do not preciscely need for inference of a model.
The methods used here require the assumptions made from the TFLM framework to
properly work.
"""