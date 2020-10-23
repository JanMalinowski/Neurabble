from .utils import two_list, create_images_pairs
import pandas as pd
import os

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIRECTORY")
    data = create_images_pairs(data_dir + "/data.pkl")
    data = data.reset_index(drop=True)
    data.to_pickle(data_dir + "/dataset.pkl")
    print("Dataset has the following shape: ", data.shape)
