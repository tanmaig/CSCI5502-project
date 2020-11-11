import pandas as pd
import argparse
from util import write_pickle_file, exists, determine_regions, read_datasets


def data_cleaning(region_wise_datasets):
    """
    Description: Function for data cleaning (missing value / null imputation, removing duplicate rows). This is not
    generalized for now. This is only for Description column as this is the only column that seems to contain null
    values in all datasets.
    :param region_wise_datasets: Dictionary containing keys as region codes (strings) and values as corresponding
    datasets (dataframes).
    :return: region_wise_datasets dictionary with cleaned datasets for each entry.
    """
    for region, dataset in region_wise_datasets.items():
        dataset.drop_duplicates(inplace=True) # Drop duplicate instances.
        dataset.reset_index(drop=True, inplace=True)
        dataset["description"].fillna("", inplace=True)
        region_wise_datasets[region] = dataset
    return region_wise_datasets


def label_generation(dataset):
    """
    Descripton: Function to generate labels for consolidated dataset using score function. Labels are defined as follows:
    1. class 0: Non popular videos; (< 100000 views)
    2. class 1: Popular videos with overwhelming bad views. (>= 100000 views, score < 0)
    3. class 2: Neutral popular videos ( >= 100000 views, score >= 0 & score < 300)
    4. class 3: Popular videos with overwhleming good views. (>= 100000, score >= 300)

    Score function is calculated as follows:
    score = (likes - 1.5*dislikes)*(comment_count / views)

    :param dataset: Dataframe containing condolidated region wise dataset.
    :return: Labels (pd.Series)
    """
    labels = dataset.apply(lambda x: (x["likes"] - 1.5*x["dislikes"])*(x["comment_count"]/x["views"]), axis=1)
    labels = labels.apply(lambda x: 1 if x < 0 else (2 if x >= 0 and x < 300 else 3))
    return labels


def feature_generation():


if __name__ == '__main__':

    # Load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path",
                        required=True,
                        help="Path to folder containing input datasets.")
    parser.add_argument("--regions",
                        default="all",
                        type=lambda x: determine_regions(x),
                        help="List of countries for which to read the dataset files.")
    parser.add_argument("--output_path",
                        default="./output",
                        type=lambda x: exists(x, "dir"),
                        help="Path to output folder.")

    args = parser.parse_args()

    dataset_path = args.dataset_path
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


    # Step 3: Consolidating US and Canada data sets into a single data set
    dataset = pd.concat([region_wise_datasets['US'],region_wise_datasets['CA']], ignore_index=True)

    # Deleting the Dictionary because it's no longer in use.
    del region_wise_datasets

    # Generate labels using score function.
    dataset["label"] = label_generation(dataset)

    # Feature generation: Generate base features and derived features.

    print(dataset)