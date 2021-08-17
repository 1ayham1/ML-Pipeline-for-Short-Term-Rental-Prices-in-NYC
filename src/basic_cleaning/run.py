#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, 
exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    '''
    updating input_artifact after basic cleaning
    '''
    
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
        
    df = pd.read_csv(artifact_local_path, low_memory=False)
   
    logger.info("basic cleaning ... removing outliers")

    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()    

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    filename = "clean_sample.csv"
    df.to_csv(filename, index=False)

    #update wandb
    artifact = wandb.Artifact(
     args.output_artifact,
     type=args.output_type,
     description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="basic data cleaning",
        fromfile_prefix_chars="@",
        )

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type", type=str, help="Type for the artifact", required=True,
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description for the artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Hueristic for minimum accepted price",
        required=True,
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Hueristic for minimum accepted price",
        required=True,
    )


    args = parser.parse_args()

    go(args)
