import argparse

import generate_cc_arrays

def main():
    '''Create cc sources with c arrays with data from each .tflite or .bmp.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'output_file',
        help=
        'generated output file. This must be a .cc or a .h'
    )
    parser.add_argument('input_file',
                        help='input bmp or tflite file to convert')
    args = parser.parse_args()
    size, cc_array = generate_cc_arrays.generate_array(args.input_file)
    generated_array_name = generate_cc_arrays.array_name(args.input_file)
    generate_cc_arrays.generate_file(args.output_file, generated_array_name, cc_array, size)


if __name__ == '__main__':
    main()
