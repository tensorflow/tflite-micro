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

flags.DEFINE_integer("epochs", 1, "number of epochs to train the model.")
flags.DEFINE_string("save_dir", "/tmp/lstm_trained_model",
                    "the directory to save the trained model.")
flags.DEFINE_boolean("save_tf_model", False,
                     "store the original unconverted tf model.")


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
        list of tuples: (data, label) pairs for train and test
    """
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  x_train = x_train / 255.  # normalize pixel values to 0-1
  x_train = x_train.astype(np.float32)
  return (x_train, y_train)


def train_lstm_model(epochs):
  """Train keras LSTM model on MNIST dataset

    Args:  epochs (int) : number of epochs to train the model

    Returns:
        tf.keras.Model: A trained keras LSTM model
    """
  model = create_model()
  x_train, y_train = get_train_data()
  callback = tf.keras.callbacks.EarlyStopping(
      monitor="val_loss",
      patience=1)  #early stop if validation loss does not drop anymore
  model.fit(x_train,
            y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=32,
            callbacks=[callback])
  return model


def save_tflite_model(model, save_dir):
  """Convert the saved TF model to tflite model, then save it as .tflite flatbuffer format

    Args:
        model (tf.keras.Model): the trained LSTM Model
        save_dir (str): directory to save the model
  """
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  save_name = "lstm.tflite"
  tflite_model = converter.convert()

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  with open(save_dir + "/" + save_name, "wb") as f:
    f.write(tflite_model)
  logging.info("Tflite model saved to %s", save_dir)


def train_save_model(save_dir, epochs=3, save_raw_model=False):
  """train and save LSTM model using keras

    Args:
        save_dir (string): save directory for the trained model
        epochs (int, optional): number of epochs to train the model. Defaults to
          3
        save_raw_model (bool): store the original unconverted tf model. Defaults
          to False
  """
  trained_model = train_lstm_model(epochs)

  # converter requires fixed shape input to work, alternative: b/225231544
  fixed_input = tf.keras.layers.Input(shape=[28, 28],
                                      batch_size=1,
                                      dtype=trained_model.inputs[0].dtype,
                                      name="fixed_input")
  fixed_output = trained_model(fixed_input)
  run_model = tf.keras.models.Model(fixed_input, fixed_output)

  if save_raw_model:
    run_model.save(save_dir, save_format="tf")
    logging.info("TF model saved to %s", save_dir)
  save_tflite_model(run_model, save_dir)


def main(_):
  train_save_model(FLAGS.save_dir, FLAGS.epochs, FLAGS.save_tf_model)


if __name__ == "__main__":
  app.run(main)
