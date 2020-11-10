import pandas as pd
import argparse
import os
from util import write_pickle_file, dir_exists


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


def read_datasets(folder_path, regions):
    """
    Description: Read dataset files (csv) of input regions from data folder containing original region wise
    datasets in csv and json formats for the following regions:
    1. CA - Canada
    2. DE - Germany
    3. FR - France
    4. GB - Great Britain
    5. IN - India
    6. JP - Japan
    7. KR - South Korea
    8. MX - Mexico
    9. RU - Russia
    10. US - United States
    :param folder_path: Path to folder containing original dataset files.
    :param regions: List of region codes (strings) for wich to read the csv dataset files.
    :return: Dictionary containing keys as region codes (strings) and values as corresponding datasets (dataframes).
    """
    # Dictionary consisting of (key, value) = (region, dataset)
    region_wise_datasets = {region: "" for region in regions}

    # Read csv files belonging to input regions from folder_path.
    filenames = [fname for fname in os.listdir(folder_path) if fname[:2] in regions and fname[-4:] == ".csv"]
    for fname in filenames:
        try:
            region_wise_datasets[fname[:2]] = pd.read_csv(dataset_path + "/" + fname, encoding="utf-8")
            print("Successfully read dataset for region \"" + fname[:2] + "\"\n")
        except Exception as e:
            print("Following exception occurred while reading dataset for region \"" + fname[:2] + "\":\n" + str(e))
            print("Continuing without reading dataset for region \"" + fname[:2] + "\"\n")
            del region_wise_datasets[fname[:2]]
            pass
    return region_wise_datasets


def data_cleaning(region_wise_datasets):
    """
    Description: Function for data cleaning (missing value / null imputation). This is not generalized for now.
    This is only for Description column as this is the only column that seems to contain null values in all datasets.
    :param region_wise_datasets: Dictionary containing keys as region codes (strings) and values as corresponding
    datasets (dataframes).
    :return: region_wise_datasets dictionary with cleaned datasets for each entry.
    """
    for region, dataset in region_wise_datasets.items():
        dataset["description"].fillna("", inplace=True)
        region_wise_datasets[region] = dataset
    return region_wise_datasets


if __name__ == '__main__':

    # Load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_path",
                        required=True,
                        help="Path to folder containing raw datasets.")
    parser.add_argument("--regions",
                        default="all",
                        type=lambda x: determine_regions(x),
                        help="List of countries for which to read the dataset files.")
    parser.add_argument("--output_path",
                        default="./output",
                        type=lambda x: dir_exists(x),
                        help="Path to output folder.")

    args = parser.parse_args()

    dataset_path = args.raw_dataset_path
    regions = args.regions
    output_path = args.output_path

    #print(dataset_path)
    #print(regions)
    #print(output_path)

    # Read dataset files from dataset_path folder for input regions.
    region_wise_datasets = read_datasets(dataset_path, regions)

    # Data pre-processing.

    # Step 1: Data cleaning
    region_wise_datasets = data_cleaning(region_wise_datasets)

    # Step 2: Save dictionary containing region wise cleaned datasets. => This can be used separately for exploratory
    # data analysis.
    write_pickle_file(region_wise_datasets, output_path + "/cleaned_data.pkl")