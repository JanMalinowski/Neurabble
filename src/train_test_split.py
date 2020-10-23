import pandas as pd
import os

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIRECTORY")
    data_file = os.environ.get("DATA_FILE")
    print(data_dir)
    data = pd.read_pickle(data_dir + data_file)

    # Shuffling the data

    data = data.sample(frac=1).reset_index(drop=True)
    train_idx = int(0.8 * data.shape[0])
    print(train_idx)
    train = data.loc[0:train_idx, :]
    test = data.loc[train_idx:, :]
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    train.to_pickle(data_dir + "/train.pkl")
    test.to_pickle(data_dir + "/test.pkl")
