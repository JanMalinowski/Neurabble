from utils import two_list, create_dataset
import pandas as pd

if __name__ == "__main__":
    data = create_dataset()
    data.to_pickle("input/dataset.pkl")
