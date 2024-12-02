
from pathlib import Path
import os
from argparse import ArgumentParser

import sys
sys.path.append(str(Path(__file__).parent.parent))

from nucleocentric import (
    get_files_in_folder
)

def main(args):
    file_path = Path(args.filepath)
    files = get_files_in_folder(file_path, args.file_extension_before)

    for file_name in files:
        filename = os.path.basename(file_name)
        filename = filename.split(".")[0]
        filename = filename.split("_")
        row = filename[0][4:5]
        col = filename[0][5:7]
        pos = filename[2][2:4]
        new_file_name = row + '-' + col +'-' + pos
        print(new_file_name)
        new_file_name = Path(file_path).joinpath(new_file_name + args.file_extension_after)
        os.replace(file_name, new_file_name)

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--filepath', type=str, required=True,
        help='Path to the files')
    parser.add_argument('--file_extension_before', type=str, required=False, default='.nd2',
        help='Extension of the files to be renamed')
    parser.add_argument('--file_extension_after', type=str, required=False, default='.nd2',
        help='Extension of the files after renaming')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)