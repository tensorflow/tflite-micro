#!/usr/bin/env python3

# compares two files to see if any lines match. Exits as a system error call if a match found.
# Inteded to compare a list of file paths that shouldn't be over-written in pull requests

import argparse
import sys

def test_files(file1, file2):
    with open(file1, "r") as first_file:
        first_paths = first_file.readlines()
        # clean whitespace and blank entires
        first_paths = [x.strip() for x in first_paths]
        first_paths = [x for x in first_paths if x]
    with open(file2, "r") as second_file:
        second_paths = second_file.readlines()
        second_paths = [x.strip() for x in second_paths]
        second_paths = [x for x in second_paths if x]
    for path in first_paths:
        if path in second_paths:
            print("fail")
            sys.exit(1)        
    sys.exit(0)
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file1", help="file to compare", default="placeholder")
    parser.add_argument("file2", help="file to compare", default="placeholder")
    args = parser.parse_args()
    test_files(args.file1, args.file2)
    
