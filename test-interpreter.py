import tflm_interpreter
import numpy as np
import matplotlib.pyplot as plt


print("----------------------------- Model --------------------------------")
# try second model

with open('tensorflow/lite/micro/integration_tests/seanet/add/add0.tflite',
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
  input_data = np.empty(1, dtype=np.int8)
  input_data[0] = i
  interpreter_wrapper.SetInputTensor(input_data)
  interpreter_wrapper.Invoke()
  output = interpreter_wrapper.GetOutputFloat()
  output_data[i] = output

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
  input_data = np.empty(1, dtype=np.int8)
  input_data[0] = i
  interpreter_wrapper.SetInputTensor(input_data)
  interpreter_wrapper.Invoke()
  output = interpreter_wrapper.GetOutputFloat()
  output_data[i] = output

print(output_data)
plt.plot(output_data)
plt.show()
