from .utils import two_list, create_images_pairs
import pandas as pd
import os

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIRECTORY")
    train_df = create_images_pairs(data_dir + "/train.pkl")
    test_df = create_images_pairs(data_dir + "/test.pkl")

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    categories = [
        "Anchor",
        "Apple",
        "Baby bottle",
        "Bomb",
        "Cactus",
        "Candle",
        "Taxi car",
        "Carrot",
        "Chess knight",
        "Clock",
        "Clown",
        "Diasy flower",
        "Dinosaur",
        "Dolphin",
        "Dragon",
        "Exclamation point",
        "Eye",
        "Fire",
        "Four leaf clover",
        "Ghost",
        "Green splats",
        "Hammer",
        "Heart",
        "Ice cube",
        "Igloo",
        "Key",
        "Ladybird (Ladybug)",
        "Light bulb",
        "Lightning bolt",
        "Lock",
        "Maple leaf",
        "Moon",
        "No Entry sign",
        "Orange scarecrow man",
        "Pencil",
        "Purple bird",
        "Purple cat",
        "Purple dobble hand man",
        "Red lips",
        "Scissors",
        "Skull and crossbones",
        "Snowflake",
        "Snowman",
        "Spider",
        "Spider’s web",
        "Sun",
        "Sunglasses",
        "Target/crosshairs",
        "Tortoise",
        "Treble clef",
        "Tree",
        "Water drip",
        "Dog",
        "Yin and Yang",
        "Zebra",
        "Question mark",
        "Cheese",
    ]
    code = list(range(0, len(categories)))
    category_mapping = dict(zip(categories, code))

    train_df["common_element"] = train_df["common_element"].map(category_mapping)
    test_df["common_element"] = test_df["common_element"].map(category_mapping)

    train_df.to_pickle(data_dir + "/train_pairs.pkl")
    test_df.to_pickle(data_dir + "/test_pairs.pkl")

    print("Trainset has the following shape: ", train_df.shape)
    print("Testset has the following shape: ", test_df.shape)
