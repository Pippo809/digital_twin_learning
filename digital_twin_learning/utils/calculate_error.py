import os, sys
import numpy as np
import argparse

import matplotlib.pyplot as plt
import pickle

def calculate_errors(file1: str, file2: str):

    with open(file1, 'rb') as f:
        nodes1, edges1 = pickle.load(f)
    with open(file2, 'rb') as f:
        nodes2, edges2 = pickle.load(f)

    success1 = np.array([e['success'] for e in edges1])
    success2 = np.array([e['success'] for e in edges2])

    chosen_ids = np.random.choice(np.arange(len(success1)), 10000)
    plt.scatter(success2[chosen_ids], success1[chosen_ids])
    plt.show()

    diff = success1 - success2

    error = np.mean(abs(diff))

    return error


def main(args):

    error = calculate_errors(args.file1, args.file2)

    print("error = ", error)

    return error

if __name__ == "__main__":
    file1 = "/home/aunagar/Personal/Thesis/code/isaacgym_anymal/results/graphs/graphmeshFULLPERCEPTIVE_SmallMesh2.pickle"
    file2 = "/home/aunagar/Personal/Thesis/code/isaacgym_anymal/results/graphs/hph2456SmallPredictedWithDGCNN1024.pickle"

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--file1", type=str, default=file1)
    argparser.add_argument("--file2", type=str, default=file2)

    args = argparser.parse_args()

    main(args)