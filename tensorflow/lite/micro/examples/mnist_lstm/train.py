# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""LSTM model training for MNIST recognition

This script is based on:
https://www.tensorflow.org/lite/models/convert/rnn
https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/experimental_new_converter/Keras_LSTM_fusion_Codelab.ipynb

Run:
`bazel build tensorflow/lite/micro/examples/mnist_lstm:train`
`bazel-bin/tensorflow/lite/micro/examples/mnist_lstm/train`
"""
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_integer("epochs", 20, "number of epochs to train the model.")
flags.DEFINE_string("save_dir", "/tmp/lstm_trained_model",
                    "the directory to save the trained model.")
flags.DEFINE_boolean("save_tf_model", False,
                     "store the original unconverted tf model.")
flags.DEFINE_boolean(
    "quantize", False,
    "convert and save the full integer (int8) quantized model.")


def create_model(units=20):
  """Create a keras LSTM model for MNIST recognition

    Args:
        units (int, optional): dimensionality of the output space for the model.
          Defaults to 20.

    Returns:
        tf.keras.Model: A Keras LSTM model
    """

  model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(28, 28), name="input"),
      tf.keras.layers.LSTM(units, return_sequences=True),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax, name="output")
  ])
  model.compile(optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
  model.summary()
  return model


def get_train_data():
  """Get MNIST train and test data

    Returns:
        tuple: (data, label) pairs for train and test
    """
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  x_train = x_train / 255.  # normalize pixel values to 0-1
  x_train = x_train.astype(np.float32)
  return (x_train, y_train)


def train_lstm_model(epochs, x_train, y_train):
  """Train keras LSTM model on MNIST dataset

    Args: epochs (int) : number of epochs to train the model
        x_train (numpy.array): list of the training data
        y_train (numpy.array): list of the corresponding array

    Returns:
        tf.keras.Model: A trained keras LSTM model
  """
  model = create_model()
  callback = tf.keras.callbacks.EarlyStopping(
      monitor="val_loss",
      patience=3)  #early stop if validation loss does not drop anymore
  model.fit(x_train,
            y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=32,
            callbacks=[callback])
  return model


def convert_quantized_tflite_model(model, x_train):
  """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format

    See
    https://www.tensorflow.org/lite/performance/post_training_integer_quant#convert_using_integer-only_quantization

    Args:
        model (tf.keras.Model): the trained LSTM Model
        x_train (numpy.array): list of the training data

    Returns:
        The converted model in serialized format.
  """

  def representative_dataset_gen(num_samples=100):
    for data in x_train[:num_samples]:
      yield [data.reshape(1, 28, 28)]

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.int8
  converter.inference_output_type = tf.int8
  converter.representative_dataset = representative_dataset_gen
  tflite_model = converter.convert()
  return tflite_model


def convert_tflite_model(model):
  """Convert the save TF model to tflite model, then save it as .tflite flatbuffer format

    Args:
        model (tf.keras.Model): the trained LSTM Model

    Returns:
        The converted model in serialized format.
  """
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  return tflite_model


def save_tflite_model(tflite_model, save_dir, model_name):
  """save the converted tflite model

  Args:
      tflite_model (binary): the converted model in serialized format.
      save_dir (str): the save directory
      model_name (str): model name to be saved
  """
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  save_path = os.path.join(save_dir, model_name)
  with open(save_path, "wb") as f:
    f.write(tflite_model)
  logging.info("Tflite model saved to %s", save_dir)


def prepare_trained_model(trained_model):
  """Fix the input of the trained model for inference

    Args:
        trained_model (tf.keras.Model): the trained LSTM model

    Returns:
        run_model (tf.keras.Model): the trained model with fixed input tensor size for inference
  """
  # TFLite converter requires fixed shape input to work, alternative: b/225231544
  fixed_input = tf.keras.layers.Input(shape=[28, 28],
                                      batch_size=1,
                                      dtype=trained_model.inputs[0].dtype,
                                      name="fixed_input")
  fixed_output = trained_model(fixed_input)
  run_model = tf.keras.models.Model(fixed_input, fixed_output)
  return run_model


def main(_):
  x_train, y_train = get_train_data()
  trained_model = train_lstm_model(FLAGS.epochs, x_train, y_train)
  run_model = prepare_trained_model(trained_model)
  # Save the tf model
  if FLAGS.save_tf_model:
    run_model.save(FLAGS.save_dir, save_format="tf")
    logging.info("TF model saved to %s", FLAGS.save_dir)

  # Convert and save the model to .tflite
  tflite_model = convert_tflite_model(run_model)
  save_tflite_model(tflite_model,
                    FLAGS.save_dir,
                    model_name="mnist_lstm.tflite")

  # Convert and save the quantized model
  if FLAGS.quantize:
    quantized_tflite_model = convert_quantized_tflite_model(run_model, x_train)
    save_tflite_model(quantized_tflite_model,
                      FLAGS.save_dir,
                      model_name="mnist_lstm_quant.tflite")


if __name__ == "__main__":
  app.run(main)
