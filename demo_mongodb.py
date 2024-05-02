import os 

#Library for making HTTP requests
import requests
#From the io module to handle bytes objects like files in the memory
from io import BytesIO
import pandas as pd
#From the dotenv import dotenv_values
from dotenv import dotenv_values

def download_and_combine_parquet_files(parquet_file_urls, hf_token):
    """
    Downloads Parquet files from the provided URLs using the given Hugging Face token and returns a combined DataFrame.

    Parameters:
    - parquet_file_urls: List of strings, URLs to the Parquet files.
    - hf_token: String, Hugging Face authorization token.

    Returns:
    - combined_df: A pandas DataFrame containing the combined data from all Parquet files.
    """

    headers = {"Authorization": f"Bearer {hf_token}"}
    print(headers["Authorization"])
    all_dataframes = []

    for parquet_file_url in parquet_file_urls:
        response = requests.get(parquet_file_url, headers=headers)
        if response.status_code == 200:
            parquet_bytes = BytesIO(response.content)
            df = pd.read_parquet(parquet_bytes)
            all_dataframes.append(df)
        else:
            print(f"Failed to download Parquet file from {parquet_file_url}: {response.status_code}")
    
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        return combined_df
    else:
        print("No dataframes to concatenate.")
        return None
    
# Commented out other parquet files below to reduce the amount of data ingested.
# One praquet file has an estimated 50,000 datapoint 
parquet_files = [
    "https://huggingface.co/api/datasets/AIatMongoDB/tech-news-embeddings/parquet/default/train/0000.parquet",
]

loaded_secrets = dotenv_values(".env")
hf_token = loaded_secrets["HF_API_KEY"]

combined_df = download_and_combine_parquet_files(parquet_files, hf_token)

# Remove the _id column from the initial dataset
combined_df = combined_df.drop(columns=['_id'])

# Convert earch numpy array in the 'embedding' column to a normal python list
combined_df['embedding'] = combined_df['embedding'].apply(lambda x: x.tolist())
