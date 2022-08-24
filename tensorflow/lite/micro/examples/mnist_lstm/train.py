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
# pylint: disable=line-too-long
"""
LSTM model training for MNIST recognition
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


def train_lstm_model():
    """Train keras LSTM model on MNIST dataset

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
              epochs=1,
              validation_split=0.92,
              batch_size=32,
              callbacks=[callback])
    return model


def save_tf_model(model, save_path):
    """Save the trained LSTM model in tensorflow format

    Args:
        model (tf.keras.Model): the trained LSTM Model
        save_path (str): path to save the model
    """

    batch_size, steps, input_size = 1, 28, 28
    run_model = tf.function(lambda x: model(x))
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([batch_size, steps, input_size], model.inputs[0].dtype))
    model.save(save_path, save_format="tf", signatures=concrete_func)
    print(f'TF model saved to {save_path}')


def save_tflite_model(model, saved_path):
    """Convert the saved TF model to tflite model, then save it as .tflite flatbuffer format

    Args:
        model (tf.keras.Model): the trained LSTM Model
        save_path (str): path to save the model
    """
    fixed_input = tf.keras.layers.Input(shape=[28, 28],
                                        batch_size=1,
                                        dtype = model.inputs[0].dtype,
                                        name='fixed_input')
    fixed_output = model(fixed_input)
    run_model = tf.keras.models.Model(
        fixed_input, fixed_output
    )  #converter requires fixed shape input to work, alternative: b/225231544
    converter = tf.lite.TFLiteConverter.from_keras_model(run_model)
    tflite_model = converter.convert()
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    with open(saved_path + '/lstm.tflite', 'wb') as f:
        f.write(tflite_model)
    print(f"Tflite model saved to {saved_path}")


def main(save_path, save_raw_model):
    """train and save LSTM model using keras

    Args:
        save_path (string): save path for the trained model
    """
    trained_model = train_lstm_model()
    if save_raw_model:
        save_tf_model(trained_model, save_path)
    save_tflite_model(trained_model, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and convert a LSTM model')
    parser.add_argument('--save_path',
                        metavar='p',
                        default='/tmp/trained_model',
                        help='the path to save the trained model')
    parser.add_argument('--save_tf_model',
                        default=False,
                        help='store the original unconverted tf model',
                        action='store_true')

    args = parser.parse_args()
    main(args.save_path, args.save_tf_model)
