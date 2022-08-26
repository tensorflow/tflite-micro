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
"""
LSTM model training for MNIST recognition

Using python3:
`python3 tensorflow/lite/micro/examples/mnist_lstm/train.py `

Using bazel:
`bazel build tensorflow/lite/micro/examples/mnist_lstm:train`
`bazel-bin/tensorflow/lite/micro/examples/mnist_lstm/train`
"""
import argparse
import os
import numpy as np
import tensorflow as tf


def create_model(units=20):
  """Create a keras LSTM model for MNIST recognition

    Args:
        units (int, optional): dimensionality of the output space for the model. Defaults to 20.

    Returns:
        tf.keras.Model: A Keras LSTM model
    """

  model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(28, 28), name='input'),
      tf.keras.layers.LSTM(units, return_sequences=True),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='output')
  ])
  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
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

    Args:
        epochs (int) : number of epochs to train the model

    Returns:
        tf.keras.Model: A trained keras LSTM model
    """
  model = create_model()
  x_train, y_train = get_train_data()
  callback = tf.keras.callbacks.EarlyStopping(
      monitor='val_loss',
      patience=1)  #early stop if validation loss does not drop anymore
  model.fit(x_train,
            y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=32,
            callbacks=[callback])
  return model


def save_tflite_model(model, save_dir, optimize):
  """Convert the saved TF model to tflite model, then save it as .tflite flatbuffer format

    Args:
        model (tf.keras.Model): the trained LSTM Model
        save_dir (str): directory to save the model
        optimize (bool): enable model conversion optimization
    """
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  save_name = 'lstm.tflite'
  if optimize:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    save_name = 'quantized_' + save_name
    print('Model conversion using tf.lite.Optimize.DEFAULT')
  tflite_model = converter.convert()

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  with open(save_dir + '/' + save_name, 'wb') as f:
    f.write(tflite_model)
  print(f"Tflite model saved to {save_dir}")


def main(epochs, save_dir, save_raw_model, optimize_conversion):
  """train and save LSTM model using keras

    Args:
        save_dir (string): save directory for the trained model
    """
  trained_model = train_lstm_model(epochs)

  # converter requires fixed shape input to work, alternative: b/225231544
  fixed_input = tf.keras.layers.Input(shape=[28, 28],
                                      batch_size=1,
                                      dtype=trained_model.inputs[0].dtype,
                                      name='fixed_input')
  fixed_output = trained_model(fixed_input)
  run_model = tf.keras.models.Model(fixed_input, fixed_output)

  if save_raw_model:
    run_model.save(save_dir, save_format="tf")
    print(f"TF model saved to {save_dir}")
  save_tflite_model(run_model, save_dir, optimize_conversion)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Train and convert a LSTM model')

  parser.add_argument('--epochs',
                      type=int,
                      default=1,
                      help='number of epochs to train the model')
  parser.add_argument('--save_dir',
                      metavar='p',
                      default='/tmp/trained_model',
                      help='the directory to save the trained model')
  parser.add_argument('--save_tf_model',
                      default=False,
                      help='store the original unconverted tf model',
                      action='store_true')
  parser.add_argument('--optimize_conversion',
                      default=False,
                      help='enable model conversion optimization',
                      action='store_true')

  args = parser.parse_args()
  main(args.epochs, args.save_dir, args.save_tf_model,
       args.optimize_conversion)
