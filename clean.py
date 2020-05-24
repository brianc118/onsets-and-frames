import argcomplete
import argparse
import os
import subprocess
import re
import shutil
import math
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Clean model to include last model state and last model major iteration state"
)
parser.add_argument("path", type=str, help="Directory path to recursively clean")
parser.add_argument(
    "--dry",
    dest="dry",
    action="store_const",
    const=True,
    default=False,
    help="Dry run",
)
parser.add_argument(
    "--keep-pt",
    dest="keep_pt",
    action="store_const",
    const=True,
    default=False,
    help="Keep .pt files",
)
argcomplete.autocomplete(parser)
args = parser.parse_args()


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        if root.count(os.path.sep) - num_sep == level:
            yield root, dirs, files


def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def getsize(start_path="."):
    root_directory = Path(start_path)
    return sum(f.stat().st_size for f in root_directory.glob("**/*"))


if __name__ == "__main__":
    total_size = 0
    for root, subdirs, files in walklevel(args.path, level=1):
        pts = [f for f in files if len(f) > 3 and f[-3:] == ".pt"]
        if pts:
            pt_val_f = [
                (int(re.split("-|\.", f)[-2]), f)
                for f in pts
                if len(re.split("-|\.", f)) >= 2 and isint(re.split("-|\.", f)[-2])
            ]
            m = max(v for (v, _) in pt_val_f)
            for i, (val, f) in enumerate(sorted(pt_val_f)):
                if val < m and not args.keep_pt:
                    size = os.path.getsize(f"{root}/{f}")
                    total_size += size
                    print("rm", f"{root}/{f}\t{math.ceil(size/1024/1024):.0f}M")
                    if not args.dry:
                        os.remove(f"{root}/{f}")
        else:
            size = getsize(f"{root}")
            print("rm -rf", f"{root}/*\t{math.ceil(size/1024/1024):.0f}M")
            if not args.dry:
                shutil.rmtree(root)
    print(f"Cleaning {math.ceil(total_size/1024/1024):.0f}M")
