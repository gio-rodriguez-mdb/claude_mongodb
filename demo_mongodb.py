import os 
import pymongo
import openai
import anthropic
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
    
def get_mongo_client(mongo_uri):
    """
    Establish connection to the MongoDB.
    """
    try:
        mdb_client = pymongo.MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return mdb_client
    except pymongo.errors.ConnectionFailure as e:
        return None
    
def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)
    
    if query_embedding is None:
        return "Invalid query or embedding genereation failed."
    
    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch":{
                "index" : "vector_index",
                "queryVector" : query_embedding,
                "path" : "embedding",
                "numCandidates" : 150, #Number of candidates matches to consider"
                "limit" : 5 #Return top 5 matches
            }
        },
        {
            "$project": {
                "_id" : 0, #Excludes the _id field
                "embedding" : 0, #Exclude the embedding field,
                "score": {
                    "$meta" : "vectorSearchScore" # Include the search score
                }
            }
        }
    ]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)

def get_embedding(text):
    """Generate an embedding for the given text using OpenAI's API"""

    #Check for valid input
    if not text or not isinstance(text, str):
        return None
    
    try:
        #Call OpenAI API to get the embedding
        embedding = openai.embeddings.create(input=text, model=EMBEDDING_MODEL, dimensions=256).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None
    
def handle_user_query(query, collection):
    get_knowledge = vector_search(query, collection)
    search_result = ''
    for result in get_knowledge:
        search_result += (
            f"Title: {result.get('title', 'N/A')}, "
            f"Company Name: {result.get('companyName', 'N/A')}, "
            f"Company URL: {result.get('companyUrl', 'N/A')}, "
            f"Date Published: {result.get('published_at', 'N/A')}, "
            f"Article URL: {result.get('url', 'N/A')}, "
            f"Description: {result.get('description', 'N/A')}, \n"
        )
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a Venture Capital Tech Analyst with access to some tech company articles and information. Use the information you are given to provide advice."},
            {"role" : "user", "content": "Answer this query: " + query + " with the following context: " + search_result}
        ]
    )
    return (response.choices[0].message.content), search_result
    
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

mongo_uri = loaded_secrets["MONGO_URI"]

if not mongo_uri:
    print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)
DB_NAME = "tech_news"
COLLECTION_NAME = "hacker_noon_tech_news"

db = mongo_client[DB_NAME]
collection = db[COLLECTION_NAME]

# Delete all the information stored in the collection
collection.delete_many({})

#Data Ingestion

combined_df_json = combined_df.to_dict(orient='records')
collection.insert_many(combined_df_json)

openai.api_key = loaded_secrets["OPENAI_API_KEY"]
EMBEDDING_MODEL = "text-embedding-3-small"

anth_client = anthropic.Client(api_key = loaded_secrets["ANTHROPIC_API_KEY"])

# Conduct query with retrieval of sources.
query = "What is the worst company to invest?, tell me why"
response, source_information = handle_user_query(query,collection)

print(f"Response: {response}")
print(f"Source Information: \\n{source_information}")