import os
import re
import tensorflow as tf
import binascii
import json


def parse_model_file(path, variable_name):
    """this will parse the hex string form a model file model_data[] = {0x18, 0x00, ....} given
    the file path and the variable name
    """
    M = []
    with open(path, "r") as fid:
        start_parsing = False
        for line in fid.readlines():
            if start_parsing:
                M.append(line.lstrip())
            if variable_name in line:
                start_parsing = True
            if "};" in line:
                start_parsing = False

    return parse_model_hex_string("".join(M).replace("};", ""))


def parse_model_hex_string(hex_string):
    """this will parse the hex string form a model file
        only pass in the string with the hex values  '0x18, 0x00...
    '"""
    hex_string.replace(",\n", "")
    return "".join([x.lstrip().rstrip()[2:4] for x in hex_string.split(",")])


def simple_name(function):
    return function.lower().replace("_", "")

def parse_all_ops(path):

    if not os.path.exists(path):
        raise Exception("cannot find all_ops_resolver.cc at path {}".format(path))

    function_list = []
    register_function = {}
    start_parsing = False
    with open(path, "r") as fid:
        for line in fid.readlines():
            if "AllOpsResolver::AllOpsResolver()" in line:
                start_parsing = True
            if "Add" in line and start_parsing:
                s = line.lstrip()[3:-4]
                function_list.extend([s])
                register_function[simple_name(s)] = line.lstrip()

    return function_list, register_function


def gen_model_functions(function_list, all_ops_code):

    M = [
        "  // Only Pull in functions that are needed by the model",
        "  static tflite::MicroMutableOpResolver<{}> resolver;".format(
            len(function_list)
        ),
    ]

    # template = "  resolver.AddBuiltin(tflite::BuiltinOperator_{0}, tflite::ops::micro::Register_{0}());"
    template = "    resolver.{0}"

    for function in sorted(list(function_list)):  # sort so always in same order
        print(function)
        M.append(template.format(all_ops_code[simple_name(function)][:-1]))

    return M


def get_name_interpreter(name):

    if "dense" in name:
        return "FULLY_CONNECTED"

    return name.upper()


def get_activation_interpreter(activation):

    activation = activation.split("/")[-1].split(':')[0]

    included_functions = [
        "RESHAPE",
        "FAKEQUANTWITHMINMAXVARS",
        "ADD",
        "MATMUL",
        "MATMUL_BIAS",
        "BIASADD",
        "INPUT",
        "TRANSPOSE",
        "IDENTITY",
        "BIAS",
        "CONV2D_BIAS",
        "SHAPE",
        "READVARIABLEOP",
        "MAXPOOL",
        "DEPTHWISE_FOLD_BIAS",
        "CONV2D_FOLD_BIAS",
        "RESOURCE",
        "AVGPOOL",
        "QUANTIZE",
        "CONST",
        "BIAS",
        "STACK",
        "STATEFULPARTITIONEDCALL",
        "SIZE",
        "FUSEDBATCHNORM",
        "CONV2D1"
    ]

    if any(x in activation.upper() for x in included_functions):
        return None

    if activation.isdigit():
        return None

    if activation:
        return activation.upper()

    return None


# TODO: This function needs to be improved. Right now it is only pulling out the ops name. Do we need to pull out activations as well?
def parse_tensorflow_binary_model(model_binary):

    tf_model = tf.lite.Interpreter(model_content=binascii.unhexlify(model_binary))

    with open("model.tflite", "wb") as out:
        out.write(binascii.unhexlify(model_binary))

    used_functions = set()
    print("Model Summary")
    for op in tf_model._get_ops_details():
        print(op)
        name = get_name_interpreter(op["op_name"])
        if name:
            used_functions.add(name)

        for index in op["inputs"]:

            activation = get_activation_interpreter(
                tf_model._get_tensor_details(index)["name"]
            )
            print("input",index, tf_model._get_tensor_details(index)["name"], activation)

            if activation:
                used_functions.add(activation)

        for index in op["outputs"]:

            activation = get_activation_interpreter(
                tf_model._get_tensor_details(index)["name"]
            )
            print("output",index, tf_model._get_tensor_details(index)["name"], activation)

            if activation:
                used_functions.add(activation)

    return used_functions


def fuzzy_match(function, micro_functions):
    return simple_name(function) in [x.lower() for x in micro_functions]


def validate_micro_functions_available(used_functions, micro_functions):

    print("AllOps Functions", micro_functions)
    for used_function in used_functions:
        print('Checking included op', simple_name(used_function))
        if not fuzzy_match(used_function, micro_functions):
            raise Exception(
                "model uses {} which is not a supported function in tf micro ".format(
                    used_function
                )
            )

    print("Operations to Include\n", used_functions)


def gen_micro_mutable_ops_resolver_add(model, all_ops_path):

    micro_functions, micro_function_code = parse_all_ops(all_ops_path)

    used_functions = parse_tensorflow_binary_model(model)

    validate_micro_functions_available(used_functions, micro_functions)

    return gen_model_functions(used_functions, micro_function_code)


def fill_micro_api_template_file(
    model=None,
    template_path="./micro_api.cc.tpl",
    output="../../../micro_api.cc",
    all_ops_path="../../../all_ops_resolver.cc",
):

    default_template = {
        "micro_mutable_ops_resolver": [
            "// All functions are included in the library",
            " static tflite::AllOpsResolver resolver;",
        ],
        "micro_mutable_ops_resolver_header": [
            '#include "tensorflow/lite/micro/all_ops_resolver.h"'
        ],
    }

    if model:
        default_template[
            "micro_mutable_ops_resolver"
        ] = gen_micro_mutable_ops_resolver_add(model, all_ops_path)
        default_template["micro_mutable_ops_resolver_header"] = [
            '#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"'
        ]

    with open(template_path, "r") as fid:
        output_str = "".join(fid.readlines())
        for key, value in default_template.items():
            output_str = re.sub(
                r"//FILL_{}\b".format(key.upper()),
                "\n".join(value),
                output_str,
            )

    with open(output, "w") as out:
        out.write(output_str)

    return default_template


def to_c_hex(tflite_model):
    hex_str = binascii.hexlify(tflite_model).decode()
    return (
        "".join(
            ["0x{}, ".format(hex_str[i : i + 2]) for i in range(0, len(hex_str), 2)]
        )[:-2],
        len(hex_str) // 2,
    )


def fill_model_template_file(
    model,
    template_path="./model.cc.tpl",
    output="../model.cc",
):

    model_str, model_length = to_c_hex(binascii.unhexlify(model))
    template = {
        "MODEL": "const unsigned char g_model[] DATA_ALIGN_ATTRIBUTE = {{{0}}};".format(
            model_str
        ),
        "MODEL_LENGTH": "const int g_model_len = {};".format(model_length),
    }

    with open(template_path, "r") as fid:
        output_str = "".join(fid.readlines())
        for key, value in template.items():
            output_str = re.sub(
                r"//FILL_{}\b".format(key.upper()),
                value,
                output_str,
            )

    with open(output, "w") as out:
        out.write(output_str)

    return template


def fill_test_data(
    test_data,
    output="../test_data.h",
):

    num_inputs = len(test_data[0])
    num_outputs = 5
    outputs = []
    outputs.append("#define MODEL_INPUTS {}".format(num_inputs))
    outputs.append("#define MODEL_OUTPUTS {}".format(num_outputs))
    outputs.append("#define TEST_DATA_LENGTH {}".format(len(test_data)))
    outputs.append(
        "float results[MODEL_OUTPUTS] ={{ {} }};".format(
            ", ".join(["0" for _ in range(num_outputs)])
        )
    )

    outputs.append("float test_data[TEST_DATA_LENGTH][MODEL_INPUTS] = {")

    for i in range(len(test_data)):
        outputs.append("{{ {} }},".format(".0f,".join([str(x) for x in test_data[i]])))

    outputs.append("};")

    with open(output, "w") as out:
        out.write("\n".join(outputs))

    return "\n".join(outputs)


def fill_class_map(
    class_map, template_path="output_handerl.cc.tpl", output="../output_handler.cc"
):

    outputs = []
    outputs.append("switch (result){")
    for index, value in class_map:
        outputs.append("case ({}):".format(index))
        outputs.append(
            "\t" + 'TF_LITE_REPORT_ERROR( error_reporter,"{}");'.format(value)
        )
    outputs.append("}")


if __name__ == "__main__":

    import sys

    if len(sys.argv) <= 1:
        fill_micro_api_template_file()

    else:
        print(sys.argv[1])
        params = json.load(open(sys.argv[1], "r"))

        if params.get("model_path", None):
            model = parse_model_file(params["model_path"], "g_model")
        elif params.get("model_binary", None):
            model = params["model_binary"]
        else:
            raise Exception("must provide either model path or model binary")

        fill_micro_api_template_file(model)
        print("generated model_api.c")

        fill_model_template_file(model)
        print("generated model.cc")

        if params.get("test_data", None):
            fill_test_data(params["test_data"])
            print("generated test_data.h")
