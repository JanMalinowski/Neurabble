# The code was taken from Abhishek Thakur's repo
# https://github.com/abhishekkrthakur/mlframework/blob/master/src/create_folds.py
import pandas as pd
from sklearn import model_selection
import os

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIRECTORY")
    train = pd.read_pickle(data_dir + "/train.pkl")
    train = train.sample(frac=1.0)
    train = train.reset_index(drop=True)
    train["kfold"] = -1

    kf = model_selection.KFold(n_splits=3)

    for (fold, (train_idx, val_idx)) in enumerate(kf.split(X=train)):
        print(len(train_idx), len(val_idx))
        train.loc[val_idx, "kfold"] = fold

    train.to_pickle(data_dir + "/train_folds.pkl")
