import pickle
import os
import shutil
import pandas as pd


def read_datasets(folder_path, regions):
    """
    Description: Read dataset files (csv) of input regions from from folder_path from the following regions:
    1. CA - Canada
    2. DE - Germany
    3. FR - France
    4. GB - Great Britain
    5. IN - India
    6. JP - Japan
    7. KR - South Korea
    8. MX - Mexico
    9. RU - Russia
    10. US.csv - United States
    :param folder_path: Path to folder containing dataset files.
    :param regions: List of region codes (strings) for wich to read the csv dataset files.
    :return: Dictionary containing keys as region codes (strings) and values as corresponding datasets (dataframes).
    """
    # Dictionary consisting of (key, value) = (region, dataset)
    region_wise_datasets = {region: "" for region in regions}

    # Read csv files belonging to input regions from folder_path.
    filenames = [fname for fname in os.listdir(folder_path) if fname[:2] in regions and fname[-4:] == ".csv"]
    for fname in filenames:
        try:
            region_wise_datasets[fname[:2]] = pd.read_csv(folder_path + "/" + fname,
                                                          encoding="utf-8")
            print("Successfully read dataset for region \"" + fname[:2] + "\"\n")
        except Exception as e:
            print("Following exception occurred while reading dataset for region \"" + fname[:2] + "\":\n" + str(e))
            print("Continuing without reading dataset for region \"" + fname[:2] + "\"\n")
            del region_wise_datasets[fname[:2]]
            pass
    return region_wise_datasets


def determine_regions(x):
    """
    Description: Function to convert regions input into list of regions.
    :param x: command line input string containing comma separated list of region codes.
    (default = "all", if no input is given).
    :return: List of region codes (strings)
    """
    if x == "all":
        x = "CA,DE,FR,GB,IN,JP,KR,MX,RU,US"
    x = x.split(",")
    return x


def exists(x, type):
    """
    Description: Create folder x; if it already exists, delete the old one first.
    :param x: Path to folder.
    :param type: Path to folder.
    :return: Path to folder.
    """
    if type == "dir":
        if os.path.exists(x):
            shutil.rmtree(x)
        os.makedirs(x)
    else:
        if not os.path.exists(x):
            raise FileNotFoundError
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