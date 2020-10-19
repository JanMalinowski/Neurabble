import pandas as pd

if __name__ == "__main__":
    data = pd.read_pickle("input/dataset.pkl")

    # Shuffling the data

    data = data.sample(frac=1).reset_index(drop=True)
    train_idx = int(0.8 * data.shape[0])
    print(train_idx)
    train = data.loc[0:train_idx, :]
    test = data.loc[train_idx:, :]
    train.to_pickle("input/train.pkl")
    test.to_pickle("input/test.pkl")
