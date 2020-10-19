# The code was taken from Abhishek Thakur's repo
# https://github.com/abhishekkrthakur/mlframework/blob/master/src/create_folds.py
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    train = pd.read_pickle("input/train.pkl")
    train["kfold"] = -1

    kf = model_selection.KFold(n_splits=2, random_state=42)

    for (fold, (train_idx, val_idx)) in enumerate(kf.split(X=train)):
        print(len(train_idx), len(val_idx))
        train.loc[val_idx, "kfold"] = fold

    train.to_pickle("input/train_folds.pkl")
