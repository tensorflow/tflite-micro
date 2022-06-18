This is a experimental tool to generate a visualization of tflite file with size info for
each field. 

The size info of each field is the raw storage size info of each field without
any flatbuffer overhead such as the offset table etc. Hence, the size info
provide a lower bound on the size of data required (such as storing it into a c
struct) instead of storing it as the tflite buffer. 

Here is how you can use a visualization of tflite file

```
from tflite_micro.tensorflow.lite.micro.python.tflite_size.src import flatbuffer_size

flatbuffer_size.convert_tflite_to_html(in_filename, out_html_file, out_json_file)
```

A sample output html is [here](./tests/gold_simple_add_model.json.html).

