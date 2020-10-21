from .utils import two_list, create_dataset
import pandas as pd
import os

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIRECTORY")
    data = create_dataset(data_dir + '/data.pkl')
    data.to_pickle(data_dir + "/dataset.pkl")
    print("Dataset has the following shape: ", data.shape)
