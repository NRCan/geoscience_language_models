# Copyright (C) 2021 ServiceNow, Inc.

import argparse
import pathlib
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--DIRS', nargs="+", help="Mlflow model dirs")

    args = parser.parse_args()

    original_dirs = [pathlib.Path(dir) for dir in args.DIRS]

    best_model_dirs = [dir / "best_model" for dir in original_dirs]

    for best_model_dir, original_dir in zip(best_model_dirs, original_dirs):

        if not best_model_dir.exists():
            pathlib.Path.mkdir(best_model_dir)

        shutil.move(original_dir / "config.json", best_model_dir / "config.json")
        shutil.move(original_dir / "pytorch_model.bin", best_model_dir / "pytorch_model.bin")
        shutil.move(original_dir / "special_tokens_map.json", best_model_dir / "special_tokens_map.json")
        shutil.move(original_dir / "tokenizer_config.json", best_model_dir / "tokenizer_config.json")
        shutil.move(original_dir / "vocab.txt", best_model_dir / "vocab.txt")


if __name__ == '__main__':
    main()
