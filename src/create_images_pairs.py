from .utils import two_list, create_images_pairs
import pandas as pd
import os
import joblib

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIRECTORY")
    train_df = create_images_pairs(data_dir + "/train.pkl")
    test_df = create_images_pairs(data_dir + "/test.pkl")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    categories = joblib.load("input/detecting_common_element/categories.pkl")

    code = list(range(0, len(categories)))
    category_mapping = dict(zip(categories, code))

    train_df["common_element"] = train_df["common_element"].map(category_mapping)
    test_df["common_element"] = test_df["common_element"].map(category_mapping)

    train_df.to_pickle(data_dir + "/train_pairs.pkl")
    test_df.to_pickle(data_dir + "/test_pairs.pkl")

    print("Trainset has the following shape: ", train_df.shape)
    print("Testset has the following shape: ", test_df.shape)
