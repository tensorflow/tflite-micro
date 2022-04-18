import tflm_interpreter
import numpy as np
import matplotlib.pyplot as plt
import sys

np.set_printoptions(threshold=sys.maxsize)

print("----------------------------- Model --------------------------------")

with open(
    'tensorflow/lite/micro/examples/memory_footprint/models/simple_add_model.tflite',
    "rb") as f:
  data = f.read()

interpreter_wrapper = tflm_interpreter.InterpreterWrapper(data)
print(interpreter_wrapper)

# Doesn't do anything right now - everything goes through InterpreterWrapper
# TODO: Figure out to keep or discard
interpreter = interpreter_wrapper.interpreter()
print(interpreter)

interpreter_wrapper.AllocateTensors()

num_steps = 100
output_data = np.empty(num_steps)
for i in range(0, num_steps):
  input_data = np.full((128, 128, 1), i, dtype=np.int8)
  interpreter_wrapper.SetInputTensor(input_data, 0)
  interpreter_wrapper.SetInputTensor(input_data, 1)
  interpreter_wrapper.Invoke()
  output = interpreter_wrapper.GetOutputTensor()
  output_data[i] = output[0, 0, 0, 0]

print(output_data)
plt.plot(output_data)
plt.show()

print("----------------------------- Model --------------------------------")

with open('tensorflow/lite/micro/examples/hello_world/hello_world.tflite',
          "rb") as f:
  data = f.read()

interpreter_wrapper = tflm_interpreter.InterpreterWrapper(data)
print(interpreter_wrapper)

# Doesn't do anything right now - everything goes through InterpreterWrapper
# TODO: Figure out to keep or discard
interpreter = interpreter_wrapper.interpreter()
print(interpreter)

interpreter_wrapper.AllocateTensors()

num_steps = 1000
output_data = np.empty(num_steps)
for i in range(0, num_steps):
  input_data = np.full((1, ), i, dtype=np.int8)
  interpreter_wrapper.SetInputTensor(input_data, 0)
  interpreter_wrapper.Invoke()
  output = interpreter_wrapper.GetOutputTensor()
  output_data[i] = output

print(output_data)
plt.plot(output_data)
plt.show()
