<!--ts-->
   * [Using Bazel](#using-bazel)
   * [Manual Setup Illustration](#manual-setup-illustration)

<!-- Added by: advaitjain, at: Fri Oct 29 11:57:14 AM PDT 2021 -->

<!--te-->

Writing and using Python scripts from the TFLM repository is currently in the
prototyping stage. As such, the instructions below are somewhat sparse and
subject to change.


* [TensorFlow Python style guide](https://www.tensorflow.org/community/contribute/code_style#python_style)


# Using Bazel

We use Bazel as our default build system for Python and the continuous
integration infrastrucutre only runs the Python unit tests via Bazel.

When using Bazel with Python, all the environment setup is handled as part of the
build.

Some example commands:
```sh
bazel test tensorflow/lite/tools:flatbuffer_utils_test
bazel build tensorflow/lite/tools:visualize

bazel-bin/tensorflow/lite/tools/visualize tensorflow/lite/micro/models/person_detect.tflite tensorflow/lite/micro/models/person_detect.tflite.html
```

# Manual Setup Illustration

For advanced users that would like to use the Python code in the TFLM repository
independent of bazel, here is one approach.

Please note that this setup is unsupported and will need users to debug various
issues on their own. It is described here for illustrative purposes only.

```sh
# The cloned tflite-micro folder needs to be renamed to tflite_micro
mv tflite-micro tflite_micro
# To set up a specific Python version, make sure `python` is pointed to the
# desired version. For example, call `python3.11 -m venv tflite_micro/venv`.
python -m venv tflite_micro/venv
echo "export PYTHONPATH=\${PYTHONPATH}:${PWD}" >> tflite_micro/venv/bin/activate
cd tflite_micro
source venv/bin/activate
pip install --upgrade pip
pip install -r third_party/python_requirements.txt

# (Optional)
pip install ipython
```

Run some tests and binaries:
```sh
python tensorflow/lite/tools/flatbuffer_utils_test.py
python tensorflow/lite/tools/visualize.py tensorflow/lite/micro/models/person_detect.tflite tensorflow/lite/micro/models/person_detect.tflite.html
```

