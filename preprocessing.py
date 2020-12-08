import pandas as pd
import numpy as np
import re
import argparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import write_pickle_file, exists, determine_regions, read_datasets


def data_cleaning(region_wise_datasets):
    """
    Description: Function for data cleaning (missing value / null imputation, removing duplicate rows and videos no longer available).
    :param region_wise_datasets: Dictionary containing keys as region codes (strings) and values as corresponding
    datasets (dataframes).
    :return: region_wise_datasets dictionary with cleaned datasets for each entry.
    """
    for region, dataset in region_wise_datasets.items():
        dataset.drop_duplicates(inplace=True) # Drop duplicate instances.
        dataset["description"].fillna("", inplace=True)
        dataset.dropna(inplace=True)  # Drop rows with nan values in any columns.
        dataset = dataset[dataset["video_error_or_removed"] == False]
        dataset.reset_index(drop=True, inplace=True)
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


def load_glove_vectors(glove_vector_path):
    """
    Description: Function to create a dictionary mapping words to glove vector embeddings.
    :param glove_vector_path: Path to file containing 25d pretrained glove vectors extracted from Twitter data.
    :return: Dictionary with reference glove vector embeddings.
    """
    embeddings_dict = {}
    with open(glove_vector_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        elements = line.split()
        word = elements[0]
        embedding = np.asarray(elements[1:], "float64")
        embeddings_dict[word] = embedding
    return embeddings_dict


def text_preprocessing(data):
    """
    Description: Function to preprocess all the NLP features.
    :param data: Row of Dataframe.
    :return: preprocessed row.
    """
    # normalize data
    data["title"] = str(data["title"]).lower()
    data["tags"] = str(data["tags"]).lower()
    data["description"] = str(data["description"]).lower()

    # remove non alphanumeric characters
    data["title"] = re.sub("[^a-zA-Z0-9]", " ", data["title"])
    data["tags"] = re.sub("[^a-zA-Z0-9]", " ", data["tags"])
    data["description"] = re.sub("[^a-zA-Z0-9]", " ", data["description"])

    # tokenization
    data["title"] = word_tokenize(data["title"])
    data["tags"] = word_tokenize(data["tags"])
    data["description"] = word_tokenize(data["description"])

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    data["title"] = [token for token in data["title"] if token not in stop_words]
    data["tags"] = [token for token in data["tags"] if token not in stop_words]
    data["description"] = [token for token in data["description"] if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer() # intialize WordNet lemmagtizer
    #data["title"] = [lemmatizer.lemmatize(w, tag) for (w, tag) in nltk.pos_tag(data["title"])]
    #data["tags"] = [lemmatizer.lemmatize(w, tag) for (w, tag) in nltk.pos_tag(data["tags"])]
    #data["description"] = [lemmatizer.lemmatize(w, tag) for (w, tag) in nltk.pos_tag(data["description"])]
    data["title"] = [lemmatizer.lemmatize(w) for w in data["title"]]
    data["tags"] = [lemmatizer.lemmatize(w) for w in data["tags"]]
    data["description"] = [lemmatizer.lemmatize(w) for w in data["description"]]

    # remove words with len <= 2
    data["title"] = [word for word in data["title"] if len(word) > 2]
    data["tags"] = [word for word in data["tags"] if len(word) > 2]
    data["description"] = [word for word in data["description"] if len(word) > 2]

    return data


def embedding(data, embeddings_dict):
    """
    Description: Convert preprocessed textual data into vector form using glove vectors. For a sentence, all word
    embeddings are aggregated.
    :param data: Row of input dataframe.
    :param embeddings_dict: Dictionary containing word to embedding mapping.
    :return: Row with textual data transformed using glove vector embeddings.
    """

    temp = np.zeros(25)
    for elem in data["title"]:
        try:
            temp = np.sum([temp, embeddings_dict[elem]], axis=0)
        except KeyError:
            pass
    data["title"] = temp

    temp = np.zeros(25)
    for elem in data["tags"]:
        try:
            temp = np.sum([temp, embeddings_dict[elem]], axis=0)
        except KeyError:
            pass
    data["tags"] = temp

    temp = np.zeros(25)
    for elem in data["description"]:
        try:
            temp = np.sum([temp, embeddings_dict[elem]], axis=0)
        except KeyError:
            pass
    data["description"] = temp
    return data


def feature_generation(dataset, embeddings_dict):
    """
    Description: Function to transform dataset columns to generate base / derived features.
    :param dataset: Dataframe containing input dataset.
    :param embeddings_dict: Dictionary containing word to embedding mapping.
    :return: Dataset (Dataframe) containing generated features.
    """
    # Feature 1: time_gap = No. of days between publishing date and trending date.
    trending_date = pd.to_datetime(dataset["trending_date"], format="%y.%d.%m").dt.date
    publish_date = pd.to_datetime(dataset["publish_time"]).dt.date
    dataset["time_gap"] = (trending_date - publish_date).dt.days

    # Feature 2, 3, 4: title, tags, description - NLP features
    try: # Exception handling - in case anyone faces any issues with tqdm installation / usage.
        tqdm.pandas()
        dataset = dataset.progress_apply(text_preprocessing, axis=1)
        dataset = dataset.progress_apply(embedding, args=(embeddings_dict,), axis=1)
    except Exception as e:
        pass
        dataset = dataset.apply(text_preprocessing, axis=1)
        dataset = dataset.apply(embedding, args=(embeddings_dict,), axis=1)

    # Feature 5: category_id : Same categories used in all regions (labelled from 1 to 44: 3 to 10 missing). (Should intercept be added?)
    # Feature 6: duration : This will be considered, if present.

    features_list = ["video_id", "time_gap", "title", "tags", "description", "category_id"]
    if "duration" in dataset.columns:
        features_list.append("duration")
    cols = features_list + ["label"]
    dataset = dataset[cols]
    return dataset


def pca(X_train, X_test, output_folder, visualize=False, trasnform=True):
    """
    Description: Perform PCA analysis on train / test sets.
    :param X_train: Training set features
    :param X_test: Test set features
    :param visualize: PCA analysis
    :param transform: Perform PCA dimensionality reduction.
    :param output_folder: Output folder to save analysis results.
    :return: Trasformed train and test sets.
    """

    # Standard scalar normalization.
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    # Apply PCA.
    pca = PCA(0.95)
    transformed_data = pca.fit_transform(X_train_scaled)
    print("Estimated no. of components = ", pca.n_components_)

    if visualize:
        plt.clf()
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.savefig(output_folder + "/PCA_analysis.png")

    if trasnform:
        X_train_transformed = transformed_data
        X_test_transformed = pca.transform(X_test_scaled)
        return X_train_transformed, X_test_transformed

    return None


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
    parser.add_argument("--glove_vector_path",
                        default="./glove.twitter.27B.25d.txt",
                        type=lambda x: exists(x, "file"),
                        help="Path to file containing pre-trained glove vectors.")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    regions = args.regions
    output_path = args.output_path
    glove_vector_path = args.glove_vector_path

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
    dataset = pd.concat([region_wise_datasets['US'], region_wise_datasets['CA']], ignore_index=True)

    # Deleting the Dictionary because it's no longer in use.
    del region_wise_datasets

    # Generate labels using score function.
    dataset["label"] = label_generation(dataset)

    # Feature generation: Generate base features and derived features.
    # For the language features, we need to transform them into features of numerical form
    # (this is done using word embeddings: GloVe vectors)
    embeddings_dict = load_glove_vectors(glove_vector_path) # Dictionary containing word, embedding pair obtained from reference file.
    dataset = feature_generation(dataset, embeddings_dict) # transform columns and generate features.

    # Going forward, ndarray would be suitable for input to most models instead of dataframe, I think.
    print(dataset.head())
    print(dataset.shape)
    print(dataset.dtypes)
    print(dataset.isna().sum())

    # Splitting title, tags and description compound features into individual features.
    for col in ["title", "tags", "description"]:
        temp = dataset[col].apply(pd.Series)
        temp.columns = [col + "_" + str(i) for i in range(0, 25)]
        dataset.drop(columns=[col], inplace=True)
        dataset = pd.concat([dataset, temp], axis=1)
        del temp

    # Convert all features into float64 dtype for consistency purposes.
    dataset[dataset.columns[~dataset.columns.isin(["video_id"])]] = dataset[
        dataset.columns[~dataset.columns.isin(["video_id"])]].apply(np.float64)

    # Splitting the dataset into the Training set and Test set.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(dataset[dataset.columns[~dataset.columns.isin(["label"])]],
                                                        dataset["label"],
                                                        test_size=0.2,
                                                        random_state=0)

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    del dataset, X_train, X_test, y_train, y_test

    # Writing to pickle files so preprocessed dataset can be directly utilized going forward.
    write_pickle_file(train, output_path + "/train.pkl")
    write_pickle_file(test, output_path + "/test.pkl")

    """# Example
    # Dimensionality reduction using PCA. - Might have to change usage if Cross-validation is used.
    train_input = train[train.columns[~train.columns.isin(["video_id"])]]
    test_input = test[test.columns[~test.columns.isin(["video_id"])]]
    train_out, test_out = pca(train_input,
                      test_input,
                      output_folder=output_path,
                      visualize=True)"""