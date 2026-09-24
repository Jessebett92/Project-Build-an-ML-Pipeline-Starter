#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import os
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# DO NOT MODIFY
def go(args):

    run = wandb.init(
        project="nyc_airbnb",
        group="cleaning",
        job_type="basic_cleaning",
        save_code=True
        )
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this

    artifact = run.use_artifact(args.input_artifact)    
    artifact_local_path = artifact.download()

    csv_filename=args.input_artifact.split(":")[0]
    csv_path = os.path.join(artifact_local_path, csv_filename)

    logger.info(f"Reading from {csv_path}")
    df = pd.read_csv(csv_path)
    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Step 6: TODO
    # Only implement this step when reaching Step 6: Pipeline Release and Updates
    # in the project.
    # Add longitude and latitude filter to allow test_proper_boundaries to pass
    # ENTER CODE HERE

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # Save the cleaned data
    df.to_csv('clean_sample.csv',index=False)

    # log the new data.
    artifact = wandb.Artifact(
     args.output_artifact,
     type=args.output_type,
     description=args.output_description,
 )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")
  
    parser.add_argument(
        "--input_artifact", 
        type = str,
        help = "Name of the input artifact to download from wandb",
        required = True
    )

    parser.add_argument(
        "--output_artifact", 
        type = str,
        help = "Name of the output artifact from wandb",
        required = True
    )

    parser.add_argument(
        "--output_type", 
        type = str,
        help = "Output artifact (type/category)",
        required = True
    )

    parser.add_argument(
        "--output_description", 
        type = str,
        help = "Output artifact description",
        required = True
    )

    parser.add_argument(
        "--min_price", 
        type = float,
        help = "Minimum price",
        required = True
    )

    parser.add_argument(
        "--max_price",
        type = float,
        help = "Maximum price",
        required = True
    )


    args = parser.parse_args()

    go(args)
