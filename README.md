# CSCI5502-project

## Initial Setup

To install python libraries, run:

```shell
$ pip install -r requirements.txt
```

## Running Instructions

For executing main pipeline, run:

```shell
$ python preprocessing.py --dataset_path ./data --regions CA,US --output_path ./output
```

For scraping duration data using Youtube Data API, run:
```shell
$ python scrape_duration.py --raw_dataset_path ./data --regions US --input_path ./input --creds_path ./creds.json
```