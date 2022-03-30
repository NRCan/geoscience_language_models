# Copyright (C) 2021 ServiceNow, Inc.
""" Count the vocabularies of the datasets in a given folder 
"""
from collections import Counter
from itertools import chain
import pathlib
from string import punctuation
import string

def countInFile(filename):
    with open(filename) as f:
        return Counter(chain.from_iterable(map(str.split, f)))

def countInFilePunctSplitLower(filename):
    with open(filename) as f:
        linewords = (line.translate(line.maketrans({p:f" {p} " for p in punctuation})).lower().split() for line in f)
        return Counter(chain.from_iterable(linewords))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATAFILE', help='datafile')
    parser.add_argument('--PROCESSING', default=None, help='datafile')
    
    args = parser.parse_args()
    
    filename = args.DATAFILE
    outfile = pathlib.Path(filename)
    outfile = outfile.parent / ("vocab_" + outfile.stem + f"_{args.PROCESSING}.csv")

    print(f'Reading {filename} with processing {args.PROCESSING}...')
    if args.PROCESSING is None or args.PROCESSING == "None":
        counter = countInFile(filename)
    elif args.PROCESSING == 'rm_punct_lower':
        counter = countInFilePunctSplitLower(filename)
    else:
        raise ValueError(f'Unknown args.PROCESSING {args.PROCESSING}')

    print(f'Writing to {outfile}...')
    with open(outfile, "w") as f:
        for k,v in counter.most_common():
            f.write(f"{k},{v}\n")
    
