from typing import List
import pandas as pd


def two_list(lst1: List[str], lst2: List[str]) -> str:
    """
    Function that return a common element of two lists
    :param lst1: first list of strings
    :param lst2: second list of strings
    """

    for elem in lst1:
        if elem in lst2:
            return elem
    raise Exception("The lists don't have a common element")


def create_images_pairs(data_path) -> pd.DataFrame:
    """
    This function takes a dataframe with columns:
    images: contains images' names
    categories: contains list of categories that the given
    image falls into

    The function returns a dataframe containing all
    images' permutations with their common element
    """
    df = pd.read_pickle(data_path)
    results = pd.DataFrame({"image1": [], "image2": [], "common_element": []})
    for index in df.index:
        image1 = df.loc[index, "images"]
        categories1 = df.loc[index, "categories"]

        if index == df.index.max():
            return results

        for index2 in df.loc[index + 1 :, :].index:
            image2 = df.loc[index2, "images"]
            categories2 = df.loc[index2, "categories"]
            # This line prevents us from comparing two different photos of the same
            # card. In addition, it also saves us troubles in case of mislabelled data.
            intersection = set(categories1).intersection(categories2)
            if len(intersection) == 1:
                common_cat = two_list(categories1, categories2)
                temp = {
                    "image1": image1,
                    "image2": image2,
                    "common_element": common_cat,
                }
                results = results.append(temp, ignore_index=True)

def read_img(path):
    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im