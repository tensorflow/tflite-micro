## Person Detection Training

In this document, you will learn how to generate a 250 KB binary classification
model to detect if a person is present in an input image or not.

## Resources

### Trained model

The trained model file (C source file `person_detect_model_data.cc`) used in
this example to run person detection on various microcontrollers is available in
[person_detection.zip](https://storage.googleapis.com/download.tensorflow.org/data/tf_lite_micro_person_data_int8_grayscale_2020_01_13.zip). This document shows you how to generate the model file.

### Dataset

We use the [Visual Wake Words dataset](https://arxiv.org/abs/1906.05721) which
contains images that belong to two classes (person or not-person). This dataset is designed to be useful for benchmarking and testing embedded computer vision, since it represents a very common task, i.e, binary classification, that we need to accomplish with tight resource constraints. We're hoping to see it drive even better models for this and similar tasks.

This is a large download (~40GB), so you'll need to make sure you have at least
100GB free on your drive to allow space for unpacking and further processing.

### Model Architecture

[MobileNets](https://arxiv.org/abs/1704.04861) are a family of efficient Convolutional Neural Networks for Mobile Vision, designed to provide good accuracy for as few weight parameters and arithmetic operations as possible.

### Compute

This model will take several hours to train on a powerful machine with GPUs and
several days with CPUs. Alternatively, we recommend using a
[Google Cloud Deep Learning VM](https://cloud.google.com/deep-learning-vm/) or
[Google Colab Pro](https://colab.research.google.com/signup) for faster training.

### Framework

We'll be training the models using the Slim library in TensorFlow 1. It is
still widely used but deprecated, so future versions of TensorFlow may not
support this approach.

Keras is the recommended interface for building models in TensorFlow 2 and
future versions, but does not support all the features we need to build the
person detection model. We hope to publish Keras instructions in the future.

## Code

### Setup

We will be running all commands from your home directory. You can place the
repository somewhere else, but you'll need to update all references to it. Now run this step initially:

```
! cd ~
```

Clone the [TensorFlow models](https://github.com/tensorflow/models) github repository:

```
! git clone https://github.com/tensorflow/models.git
```

Specifically, we will be using `~/models/research/slim` a [library](https://github.com/tensorflow/models/tree/master/research/slim) for defining, training and evaluating models. However, in order to use it, you'll need to make sure its modules can be found by Python, and
install one dependency. Here's how to do this in an iPython notebook:

```
! pip install contextlib2
import os
new_python_path = (os.environ.get("PYTHONPATH") or '') + ":models/research/slim"
%env PYTHONPATH=$new_python_path
```

### Download the Dataset

The [Visual Wake Words dataset](https://arxiv.org/abs/1906.05721) contains images which belong to two classes: person (labelled as 1) and not-person (labelled as 0) and it is derived from the [COCO dataset](http://cocodataset.org/#explore) containing 80 categories (eg: cat, dog, umbrella, etc). You can download the dataset by running this script:

```
! python models/research/slim/download_and_convert_data.py \
--logtostderr \
--dataset_name=visualwakewords \
--dataset_dir=person_detection_dataset \
--foreground_class_of_interest='person' \
--small_object_area_threshold=0.005
```

This will take several minutes (~20 minutes or more) so you may have to wait for a while before you proceed onto the next part.  When it's done, you'll have a set of TFRecords in the  `person_detection_dataset/` directory holding the labeled image information.

The script takes a long time as the COCO dataset does not have a label for each image, instead each image comes with a list of labelled bounding boxes. To create the Visual WakeWords dataset, we loop over every image and its bounding boxes and if an image has at least one bounding box labelled as 'person' with an area greater than 0.5% of the area of the image, then the entire image is labelled as "person", otherwise it is labelled as "non-person".

### Train the model

```
! python models/research/slim/train_image_classifier.py \
    --alsologtostderr \
    --dataset_name=visualwakewords \
    --dataset_dir=person_detection_dataset \
    --dataset_split_name=train \
    --train_image_size=96 \
    --use_grayscale=True \
    --preprocessing_name=mobilenet_v1 \
    --model_name=mobilenet_v1_025 \
    --train_dir=person_detection_train \
    --save_summaries_secs=300 \
    --learning_rate=0.045 \
    --label_smoothing=0.1 \
    --learning_rate_decay_factor=0.98 \
    --num_epochs_per_decay=2.5 \
    --moving_average_decay=0.9999 \
    --batch_size=96 \
    --max_number_of_steps=1000000
```

This will take a couple of days on a single-GPU v100 instance to complete all
one-million steps, but you should be able to get a fairly accurate model after
a few hours if you want to experiment early.

- `--dataset_dir` parameter should match the one where you saved the
TFRecords from the Visual Wake Words build script from the previous step.
- `--preprocessing_name` controls how input images are modified before they're
fed into the model. It reduces each image to the size specified by `--train_image_size` (here 96), convert them to grayscale using `--use_grayscale=True` which is compatible with the monochrome [HM01B0](https://himax.com.tw/products/cmos-image-sensor/image-sensors/hm01b0/) camera we're using on the SparkFun Edge board and scale the pixel values from 0 to 255 integers into -1.0 to +1.0 floating point numbers (which will be [quantized](https://en.wikipedia.org/wiki/Quantization) after training).
- `--model_name` is the model architecture we'll be using; here it's `mobilenet_v1_025`. The 'mobilenet_v1' prefix tells the script to use the first version of MobileNet. We use V1  as it uses the least amount of RAM for its intermediate activation buffers compared to later versions. The '025' is the depth multiplier, which reduces the number of weight parameters. This low setting ensures the model fits within 250KB of Flash.
- `--train_dir` will contain the trained checkpoints and summaries.
- The `--learning_rate`, `--label_smoothing`, `--learning_rate_decay_factor`,
`--num_epochs_per_decay`, `--moving_average_decay` and `--batch_size` are all
parameters that control how weights are updated during the training
process. Training deep networks is still a bit of a dark art, so these exact
values we found through experimentation for this particular model. You can try
tweaking them to speed up training or gain a small boost in accuracy, but we
can't give much guidance for how to make those changes, and it's easy to get
combinations where the training accuracy never converges.
- The `--max_number_of_steps` defines how long the training should continue.
There's no good way to figure out this threshold in advance, you have to
experiment to tell when the accuracy of the model is no longer improving to
tell when to cut it off. In our case we default to a million steps, since with
this particular model we know that's a good point to stop.

Once you start the script, you should see output that looks something like this:

```
INFO:tensorflow:global step 4670: loss = 0.7112 (0.251 sec/step)
I0928 00:16:21.774756 140518023943616 learning.py:507] global step 4670: loss =
0.7112 (0.251 sec/step)
INFO:tensorflow:global step 4680: loss = 0.6596 (0.227 sec/step)
I0928 00:16:24.365901 140518023943616 learning.py:507] global step 4680: loss =
0.6596 (0.227 sec/step)
```

Don't worry about the line duplication, this is just a side-effect of the way
TensorFlow log printing interacts with Python. Each line has two key bits of
information about the training process.
1. The `global step` is a count of how far
through the training we are. Since we've set the limit as a million steps, in
this case we're nearly five percent complete. The steps per second estimate is
also useful, since you can use it to estimate a rough duration for the whole
training process. In this case, we're completing about four steps a second, so
a million steps will take about 70 hours, or three days.
2. The `loss` is a measure of how close the partially-trained model's predictions are to the correct values, and lower values are *better*. This will show a lot of variation but should on an average decrease during training if the model is learning. This kind of variation is a lot easier to see in a graph, which is one of the main reasons to try TensorBoard.

#### TensorBoard

TensorBoard is a web application that lets you view data visualizations from
TensorFlow training sessions. You can start Tensorboard using the command line:
Run: `tensorboard --logdir person_detection_train`. Go to the URL it provides.

It may take a little while for the graphs to have anything useful in them, since
the script only saves summaries every five minutes (or 300 seconds). The most important graph is
called `clone_loss` and this shows the progression of the same loss value
that's displayed on the logging output. It fluctuates a lot, but the
overall trend is downwards over time. If you don't see this sort of progression
after a few hours of training, it's a sign that your model isn't
converging to a good solution, and you may need to debug what's going wrong
either with your dataset or the training parameters.

TensorBoard defaults to the 'Scalars' tab when it opens, but the other section
that can be useful during training is 'Images'. This shows a
random selection of the pictures the model is currently being trained on,
including any distortions and other preprocessing. This information isn't as
essential as the loss graphs, but it can be useful to ensure the dataset is what
you expect, and it is interesting to see the examples updating as training
progresses.

### Evaluate the model

(You don't need to wait until the model is fully trained, you
can check the accuracy of any checkpoints in the `--train_dir` folder.)

```
! python models/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --dataset_name=visualwakewords \
    --dataset_dir=person_detection_train \
    --dataset_split_name=val \
    --eval_image_size=96 \
    --use_grayscale=True \
    --preprocessing_name=mobilenet_v1 \
    --model_name=mobilenet_v1_025 \
    --train_dir=person_detection_train \
    --checkpoint_path=person_detection_train/model.ckpt-123456
```

You'll need to make sure that `--checkpoint_path` is pointing to a valid set of
checkpoint data. Checkpoints are stored in three separate files, so the value
should be their common prefix. For example if you have a checkpoint file called
'model.ckpt-5179.data-00000-of-00001', the prefix would be 'model.ckpt-5179'.
The script should produce output that looks something like this:

```
INFO:tensorflow:Evaluation [406/406]
I0929 22:52:59.936022 140225887045056 evaluation.py:167] Evaluation [406/406]
eval/Accuracy[0.717438412]eval/Recall_5[1]
```

The important number here is the accuracy. It shows the proportion of the
images that were classified correctly, which is 72% in this case, after
converting to a percentage. If you follow the example script, you should expect
a fully-trained model to achieve an accuracy of around 84% after one million
steps, and show a loss of around 0.4.

### Convert the TF model to a TF Lite model for Inference

When the model has trained to an accuracy you're happy with, you'll need to
convert the results from the TensorFlow training environment into a form you
can run on an embedded device. As we've seen in previous chapters, this can be
a complex process, and tf.slim adds a few of its own wrinkles too.

#### Generate the model graph

Slim generates the architecture from the `model_name` every time one of its
scripts is run, so for a model to be used outside of Slim it needs to be saved
in a common format. We're going to use the GraphDef protobuf serialization
format, since that's understood by both Slim and the rest of TensorFlow. This contains the layout of the operations in the model, but doesn't yet have any of the weight data.

```
! python models/research/slim/export_inference_graph.py \
    --alsologtostderr \
    --dataset_name=visualwakewords \
    --image_size=96 \
    --use_grayscale=True \
    --model_name=mobilenet_v1_025 \
    --output_file=person_detection_graph.pb
```

You should have a new 'person_detection_graph.pb' file in
your home folder.

#### Generate the frozen model graph (combine model graph and trained weights)

The process of storing the trained weights together with the operation graph is
known as freezing. This converts all of the variables in the graph to
constants, after loading their values from a checkpoint file. The command below
uses a checkpoint from the millionth training step, but you can supply any
valid checkpoint path. The graph freezing script is stored inside the main
TensorFlow repository, so we have to download this from GitHub before running
this command.

```
! git clone https://github.com/tensorflow/tensorflow
! python tensorflow/tensorflow/python/tools/freeze_graph.py \
--input_graph=person_detection_graph.pb \
--input_checkpoint=person_detection_train/model.ckpt-1000000 \
--input_binary=true \
--output_node_names=MobilenetV1/Predictions/Reshape_1 \
--output_graph=person_detection_frozen_graph.pb
```

After this, you should see a file called `person_detection_frozen_graph.pb`

#### Generate the TensorFlow Lite File with Quantization

[Quantization](https://en.wikipedia.org/wiki/Quantization) is a tricky and involved process, and it's still very much an
active area of research, so taking the float graph that we've trained so far
and converting it down to eight bit takes quite a bit of code. You can find
more of an explanation of what quantization is and how it works in the chapter
on latency optimization, but here we'll show you how to use it with the model
we've trained. The majority of the code is preparing example images to feed
into the trained network, so that the ranges of the activation layers in
typical use can be measured. We rely on the TFLiteConverter class to handle the
quantization and conversion into the TensorFlow Lite FlatBuffer file that we
need for the on-device inference engine.

```
import tensorflow.compat.v1 as tf
import io
import PIL
import numpy as np

def representative_dataset_gen():

  record_iterator =
tf.python_io.tf_record_iterator(path='person_detection_dataset/val.record-00000-of-00010')

  for _ in range(250):
	string_record = next(record_iterator)
    example = tf.train.Example()
    example.ParseFromString(string_record)
    image_stream =
io.BytesIO(example.features.feature['image/encoded'].bytes_list.value[0])
    image = PIL.Image.open(image_stream)
    image = image.resize((96, 96))
    image = image.convert('L')
    array = np.array(image)
    array = np.expand_dims(array, axis=2)
    array = np.expand_dims(array, axis=0)
    array = ((array / 127.5) - 1.0).astype(np.float32)
    yield([array])

converter =
tf.lite.TFLiteConverter.from_frozen_graph('person_detection_frozen_graph.pb',
['input'], ['MobilenetV1/Predictions/Reshape_1'])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
open("person_detection_model.tflite", "wb").write(tflite_quant_model)
```

#### Generate the C source file

The converter writes out a file, but most embedded devices don't have a file
system. To access the serialized data from our program, we have to compile it
into the executable and store it in Flash. The easiest way to do that is to
convert the file into a C data array.

```
# Install xxd if it is not available
! apt-get -qq install xxd
# Save the file as a C source file
! xxd -i person_detection_model.tflite > person_detect_model_data.cc
```

You can now replace the existing `person_detect_model_data.cc` file with the
version you've trained, and be able to run your own model on embedded devices.

## Other resources
### Training for a different category

To customize your model you can update the `foreground_class_of_interest` to one of the 80 categories from the COCO dataset and adjust the threshold by modifying `small_object_area_threshold`. Here's an example that looks for cars:

```
! python models/research/slim/download_and_convert_data.py \
--logtostderr \
--dataset_name=visualwakewords \
--dataset_dir=car_dataset \
--foreground_class_of_interest='car' \
--small_object_area_threshold=0.005
```

If the kind of object you're interested in isn't present in MS-COCO, you may be
able to use transfer learning to help you train on a custom dataset you've
gathered, even if it's much smaller. We don't have an example of this
yet, but we hope to share one soon.

### Understanding the Model Architecture

[MobileNets](https://arxiv.org/abs/1704.04861) are a family of architectures
designed to provide good accuracy for as few weight parameters and arithmetic
operations as possible. There are now multiple versions, but in our case we're
using the original v1 since it required the smallest amount of RAM at runtime.
The core concept behind the architecture is depthwise separable convolution.
This is a variant of classical two-dimensional convolutions that works in a
much more efficient way, without sacrificing very much accuracy. Regular
convolution calculates an output value based on applying a filter of a
particular size across all channels of the input. This means the number of
calculations involved in each output is width of the filter multiplied by
height, multiplied by the number of input channels. Depthwise convolution
breaks this large calculation into separate parts. First each input channel is
filtered by one or more rectangular filters to produce intermediate values.
These values are then combined using pointwise convolutions. This dramatically
reduces the number of calculations needed, and in practice produces similar
results to regular convolution.

MobileNet v1 is a stack of 14 of these depthwise separable convolution layers
with an average pool, then a fully-connected layer followed by a softmax at the
end. We've specified a 'width multiplier' of 0.25, which has the effect of
reducing the number of computations down to around 60 million per inference, by
shrinking the number of channels in each activation layer by 75% compared to
the standard model. In essence it's very similar to a normal convolutional
neural network in operation, with each layer learning patterns in the input.
Earlier layers act more like edge recognition filters, spotting low-level
structure in the image, and later layers synthesize that information into more
abstract patterns that help with the final object classification.



