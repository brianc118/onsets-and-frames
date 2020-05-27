import argparse
import os
import pickle
import re
from evaluate import *


def isint(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_root", type=str)
    parser.add_argument("dataset", nargs="?", default="MAPS")
    parser.add_argument("dataset_group", nargs="?", default=None)
    parser.add_argument("dataset_path", nargs="?", type=str, default=None)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--sequence-length", default=None, type=int)
    parser.add_argument("--onset-threshold", default=0.5, type=float)
    parser.add_argument("--frame-threshold", default=0.5, type=float)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    model_files = []

    for f in os.listdir(args.model_root):
        if (
            len(f) > 3
            and f[-3:] == ".pt"
            and isint(re.split("-|\.", f)[-2])
            and int(re.split("-|\.", f)[-2])
        ):
            model_files.append(os.path.join(args.model_root, f))

    model_files.sort()
    iterations = [int(re.split("-|\.", f)[-2]) for f in model_files]
    print(model_files, iterations)
    with torch.no_grad():
        res = dict(
            iterations=iterations,
            res=evaluate_files(
                model_files,
                args.dataset,
                args.dataset_group,
                args.dataset_path,
                args.save_path,
                args.sequence_length,
                args.onset_threshold,
                args.frame_threshold,
                args.device,
            ),
        )
        pickle.dump(res, open(f"{args.model_root}.p", "wb"))
        print(res)
