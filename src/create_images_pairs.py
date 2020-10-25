from .utils import two_list, create_images_pairs
import pandas as pd
import os

if __name__ == "__main__":
    data_dir = os.environ.get("DATA_DIRECTORY")
    data = create_images_pairs(data_dir + "/data.pkl")
    data = data.reset_index(drop=True)
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
    data["common_element"] = data["common_element"].map(category_mapping)
    data.to_pickle(data_dir + "/dataset.pkl")
    print("Dataset has the following shape: ", data.shape)
