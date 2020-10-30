export DATA_DIRECTORY=input/detecting_common_element/
export DATA_FILE=data.pkl

python -m src.train_test_split
python -m src.create_images_pairs
python -m src.create_folds
