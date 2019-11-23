import numpy as np
import pandas as pd
import pickle
import os


def preprocess(fpath):
    data = pd.read_csv(fpath, names=["asu", "lba", "size", "opcode", "timestamp"])
    data = data["lba"]
    data.drop(np.argwhere(data.isna()), inplace=True)
    data.drop(np.argwhere(data < 0)[:, 0], inplace=True)
    data = data.astype(np.int64)

    save_path = fpath[:-4] + ".pkl"
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    return


def load(trace):
    fpath = "./data/" + trace + ".pkl"
    if not os.path.exists(fpath):
        preprocess(fpath[:-4]+".csv")
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    files = ["Financial1", "Financial2", "WebSearch1", "WebSearch2", "WebSearch3"]
    for file in ["./data/"+file+".pkl" for file in files]:
        preprocess(file[:-4]+".csv")
