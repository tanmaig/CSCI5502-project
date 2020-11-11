import requests
import argparse
import json
from util import determine_regions, exists, read_datasets
import isodate
from tqdm import tqdm
from nested_lookup import nested_lookup



def scrape_duration(video_id, api_key):
    """
    Description: Function to scrape data using Youtube Data API.
    :param video_id: input youtube video_id value.
    :param api_key: Youtube API Key.
    :return: Video duration in seconds.
    """
    search_url = "https://www.googleapis.com/youtube/v3/videos?id="+video_id+"&key="+api_key+"&part=contentDetails"
    response = requests.get(search_url)
    content = json.loads(response.text)
    try:
        #duration = content
        duration = nested_lookup("duration", content)
        #duration = content["items"][0]["contentDetails"]["duration"]
        #duration = isodate.parse_duration(duration).total_seconds()
    except Exception as e:
        pass
        duration = None
    return duration


if __name__ == "__main__":
    # Load command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_path",
                        required=True,
                        help="Path to folder containing raw datasets.")
    parser.add_argument("--regions",
                        default="all",
                        type=lambda x: determine_regions(x),
                        help="List of countries for which to read the dataset files.")
    parser.add_argument("--input_path",
                        default="./input",
                        type=lambda x: exists(x, type="dir"),
                        help="Path to folder for storing modified input datasets.")
    parser.add_argument("--creds_path",
                        default="./creds.json",
                        type=lambda x: exists(x, type="file"),
                        help="Path to credentials file.")

    args = parser.parse_args()

    dataset_path = args.raw_dataset_path
    regions = args.regions
    input_path = args.input_path
    creds_path = args.creds_path

    region_wise_datasets = read_datasets(dataset_path, regions)

    with open(creds_path, "r") as f:
        data = json.load(f)
    api_key = data["api_key"]

    for region, dataset in region_wise_datasets.items():
        unique_videos = dataset[["video_id"]]
        unique_videos = unique_videos.head(1)
        unique_videos.drop_duplicates(inplace=True)
        tqdm.pandas()
        unique_videos["duration"] = unique_videos["video_id"].progress_apply(scrape_duration, args=(api_key,))
        print(unique_videos["duration"].isna().sum())
        print("Writing results for region \"" + region + "\" to file..")
        dataset = dataset.merge(unique_videos, how='left', on='video_id')
        dataset.to_csv(input_path + "/" + region + ".csv", index=False)