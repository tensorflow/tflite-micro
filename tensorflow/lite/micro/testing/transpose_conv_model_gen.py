from absl import app
import numpy as np
import tensorflow as tf



def convert_float_model(model: tf.keras.Model) -> bytes:
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  return converter.convert()


def get_model() -> tf.keras.Model:
  test = tf.keras.Sequential(
      [
          tf.keras.layers.Conv2DTranspose(
              1,
              (2, 2), strides=(1, 1), padding="valid", input_shape=(2, 2, 1)),
      ],
      name="test_model",
  )
  return test


def dataset_example(num_samples: int = 100):
  np.random.seed(0)  #Seed the random number generator num_samp / sample_size
  for _ in range(num_samples):
    yield [np.random.uniform(-1, 1, (1, 2, 2)).astype(np.float32)]


def generate_model():
  # Get Keras model
  keras_floating_model = get_model()
  weights = [np.asarray([[[[1]], [[2]]], [[[2]], [[1]]]]), np.asarray([0])]
  keras_floating_model.layers[0].set_weights(weights)
  input_raw = np.array([[55, 52], [57, 50]])

  input_reshape = input_raw.reshape(1, 2, 2, 1)
  keras_output = keras_floating_model.predict(input_reshape)
  keras_output = keras_output.reshape(3, 3)

  print("\nrow 0 filter:")
  print(keras_floating_model.layers[0].get_weights()[0][0][0][0][0])
  print(keras_floating_model.layers[0].get_weights()[0][0][1][0][0])
  print("\nrow 1 filter:")
  print(keras_floating_model.layers[0].get_weights()[0][1][0][0][0])
  print(keras_floating_model.layers[0].get_weights()[0][1][1][0][0])

  print("\n input matrix:")
  print(str(input_raw[0][0]) + " | " + str(input_raw[0][1]))
  print(str(input_raw[1][0]) + " | " + str(input_raw[1][1]))

  print("\nkernel/filter matrix:")
  print(
      str(keras_floating_model.layers[0].get_weights()[0][0][0][0][0]) + " | " +
      str(keras_floating_model.layers[0].get_weights()[0][0][1][0][0]))
  print(
      str(keras_floating_model.layers[0].get_weights()[0][1][0][0][0]) + " | " +
      str(keras_floating_model.layers[0].get_weights()[0][1][1][0][0]))

  print("\n kernel/filter shape: ")
  print(keras_floating_model.layers[0].get_weights()[0].shape)

  print("\nKeras Conv2D Output Matrix Reshaped 3x3:\n {}".format(keras_output))

  # Float model
  tflite_float_model = convert_float_model(keras_floating_model)

  with open("/tmp/test_git.tflite", "wb") as f:
    f.write(tflite_float_model)


def main(_):
  generate_model()


if __name__ == "__main__":
  app.run(main)

