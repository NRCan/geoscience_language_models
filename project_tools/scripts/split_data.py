# Copyright (C) 2021 ServiceNow, Inc.
#
# Split a dataset into train/test split for BERT training
# 
import os
import argparse
import pathlib

from sklearn.model_selection import train_test_split

def read_txt(file):
    with open(file, "r") as f:
        lines = f.readlines()
    non_empty_lines = [line.strip() for line in lines if len(line.strip()) > 0]
    return non_empty_lines

def write_txt(file, data):
    split_dir = os.path.splitext(file)[0]
    with open(split_dir + "/train.txt", "w") as f:
        f.write('\n'.join(train) + '\n')

if __name__ == "__main__":
    """
    python split_data.py --INPUT_FILE /nrcan_p2/data/03_primary/v4/all_text_SIMPLE_PIPELINE_BERT_3_POSTPIPE_BERT_SPACY_2_dA_full_v1.txt --OUTPUT_DIR /nrcan_p2/data/04_feature/v4/all_text_SIMPLE_PIPELINE_BERT_3_POSTPIPE_BERT_SPACY_2_dA_full_v1
    """
    
    parser = argparse.ArgumentParser()
    
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--INPUT_FILE', type=str, help='txt file to be split', required=True)
    requiredNamed.add_argument('--OUTPUT_DIR', type=str, help='output dir to save train and validation sets', required=True)
    parser.add_argument('--TRAIN_PERC', type=float, help='training set percentage, must be between (0, 1]')
    parser.add_argument('--SHUFFLE', dest='SHUFFLE', action='store_true',
                        help='whether or not to shuffle the examples before split')
    parser.add_argument('--OVERWRITE', dest='OVERWRITE', action='store_true',
                        help='whether or not to overwrite the output files if they already exist')
        
    parser.set_defaults(TRAIN_PERC=0.8)
    parser.set_defaults(SHUFFLE=True)
    parser.set_defaults(OVERWTITE=False)
    args = parser.parse_args()
    
    if args.TRAIN_PERC <= 0 or args.TRAIN_PERC > 1:
        raise ValueError("TRAIN_PERC must be between (0, 1]")
    
    if not os.path.exists(args.OUTPUT_DIR):
        os.makedirs(args.OUTPUT_DIR)
    if (len(os.listdir(args.OUTPUT_DIR)) > 0) and (not args.OVERWTITE):
        raise FileExistsError(f"{args.OUTPUT_DIR} is not empty! Set OVERWTITE to True to overwrite.")
     
    print(f"Reading data from {args.INPUT_FILE}")
    lines = read_txt(args.INPUT_FILE)
    
    print("Splitting data")
    train, validation = train_test_split(lines, shuffle=args.SHUFFLE, test_size=1-args.TRAIN_PERC)
    
    print(f"Writing to {args.OUTPUT_DIR}")
    with open(pathlib.Path(args.OUTPUT_DIR) / "train.txt", "w") as f:
        f.write('\n'.join(train) + '\n')
        
    with open(pathlib.Path(args.OUTPUT_DIR) / "validation.txt", "w") as f:
        f.write('\n'.join(validation) + '\n')
    
    