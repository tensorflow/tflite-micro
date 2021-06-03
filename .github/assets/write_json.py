#!/usr/bin/env python3

# writes a key/value json pair to an arbitrary file
# defaults to writing a random number (1-1000) to checkin_test.json in the current directory

import os
import random
import json
import argparse

def write_json(output_file, values):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    json_dict = {values[0]: values[1]}
    with open(output_file, 'w') as outfile:
        json.dump(json_dict, outfile)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help="output file path", default="./checkin_test.json")
    parser.add_argument('-v', '--values', help="key value pair, must be two strings", nargs=2, default=['random', '22'])
    args = parser.parse_args()
    if args.values[0] == 'random': args.values[1] = random.randrange(1000)
    write_json(args.file, args.values)    
    

