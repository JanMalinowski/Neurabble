export DATA_DIRECTORY=input/detecting_common_element/
python -m src.create_dataset

export DATA_FILE=dataset.pkl
python -m src.train_test_split
python -m src.create_folds

export DATA_DIRECTORY=input/multiclass_classification/
export DATA_FILE=data.pkl
python -m src.train_test_split
python -m src.create_folds