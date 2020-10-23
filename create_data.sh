export DATA_DIRECTORY=input/detecting_common_element/
python -m src.create_images_pairs

export DATA_FILE=dataset.pkl
python -m src.train_test_split
python -m src.create_folds
