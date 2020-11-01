import pandas as pd
import os

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIRECTORY")
    data_file = os.environ.get("DATA_FILE")
    print(data_dir)
    data_df = pd.read_pickle(data_dir + data_file)
    test_df = pd.DataFrame()

    # Creating two non-overlapping datasets for training and validation
    # by taking all pictures of 15 cards and putting them in the test data frame
    for i in range(15):
        temp = data_df.loc[
            data_df["categories"].apply(
                lambda x: set(x) == set(data_df["categories"][i])
            ),
            :,
        ]
        test_df = pd.concat([test_df, temp])

    data_df = data_df.drop(test_df.index.values, axis=0).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Shuffling the data
    data_df = data_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    print(data_df.shape)
    print(test_df.shape)
    data_df.to_pickle(data_dir + "/train.pkl")
    test_df.to_pickle(data_dir + "/test.pkl")
