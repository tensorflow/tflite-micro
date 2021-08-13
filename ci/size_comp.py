#!/usr/bin/env python3

import argparse
import json
import sys

def berkeley_size_format_to_dict(berkeley_size_format):
    lines = berkeley_size_format.split('\n')
    labels = lines[0].split()
    values = lines[1].split()
    outdict = {labels[i]: values[i] for i in range(len(labels) -2 )}
    return(outdict)

def json_to_dict(some_json):
    outdict = json.loads(some_json)
    return(outdict)

def file_to_dict(a_file):
    with open(a_file) as the_file:
        contents=the_file.read()
    if contents[0]=="{":
        retdict = json_to_dict(contents)
    else:
        retdict = berkeley_size_format_to_dict(contents)

    return(retdict)
    
def compare_val_in_files(old_file, new_file, val='bss'):
    old_dict = file_to_dict(old_file)
    new_dict = file_to_dict(new_file)

    if int(new_dict[val]) > int(old_dict[val]):
        print(val, " larger than previous value")
        print("old: ", old_dict[val])
        print("new: ", new_dict[val])
        print("=====Check failed=====")
        sys.exit(1)

    print(val)
    print("old: ", old_dict[val])
    print("new: ", new_dict[val])
    print("Check Passed")

    return()

def berkeley_size_format_to_json_file(input_file, output_file):
    output_dict = file_to_dict(input_file)
    with open(output_file, 'w') as outfile:
        json.dump(output_dict, outfile)

    return()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--transform", help="transform a berkeley_size_format file to a json file", nargs=2)
    parser.add_argument("-c","--compare", help="compare value in old file to new file", nargs=2)
    parser.add_argument("-v","--value", default="bss", help="value to be compared")
    args = parser.parse_args()

    if args.transform:
        berkeley_size_format_to_json_file(args.transform[0], args.transform[1])

    if args.compare:
        compare_val_in_files(args.compare[0], args.compare[1], args.value)
              

