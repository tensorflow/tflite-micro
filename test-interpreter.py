import tflm_interpreter
import numpy as np
import matplotlib.pyplot as plt

with open('tensorflow/lite/micro/examples/hello_world/hello_world.tflite',"rb") as f:
    data = f.read()

interpreter_wrapper = tflm_interpreter.InterpreterWrapper(data)
print(interpreter_wrapper)

interpreter = interpreter_wrapper.interpreter()
print(interpreter)

interpreter_wrapper.AllocateTensors()

input_data = 1
print(input_data)

kXrange = 2 * 3.14159265359

num_steps = 1000
output_data = np.empty(num_steps)
for i in range(0,num_steps):
    interpreter_wrapper.SetInputFloat(i/num_steps*kXrange)
    interpreter_wrapper.Invoke()
    output = interpreter_wrapper.GetOutputFloat()
    output_data[i] = output

print(output_data)
plt.plot(output_data)
plt.show()

# try second model
