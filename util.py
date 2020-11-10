import pickle
import os
import shutil


def dir_exists(x):
    """
    Description: Create folder x; if it already exists, delete the old one first.
    :param x: Path to folder.
    :param type: Path to folder.
    :return: Path to folder.
    """
    if os.path.exists(x):
        shutil.rmtree(x)
    os.makedirs(x)
    return x


def write_pickle_file(obj, file_path):
    """
    Description: Write a python object / data structure to pickle file.
    :param obj: Python object to be pickled.
    :param file_path: Path to pickle file.
    :return: None
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)
    return None


def read_pickle_file(file_path):
    """
    Description: Read python object from pickle file
    :param file_path: Path to pickle file.
    :return: Unpickled python object
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f)
    return obj