from fastapi import FastAPI

#Weaviate related imports
import weaviate
from weaviate.classes.init import Auth
import os
from weaviate.classes.config import Configure
import weaviate.client

#SentenceTransformer
from sentence_transformers import SentenceTransformer, util
import numpy as np

from Utilities import * 

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


app = FastAPI()

# Load the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose your preferred model

#Login to Weaviate
os.environ["WCD_URL"] = "https://pebiftvytkqbm0mx5m52qw.c0.asia-southeast1.gcp.weaviate.cloud"
os.environ["WCD_API_KEY"] = "wJYMiA1eUYI8b911rkNZN2n5fyr1LzOwV2UP"

# Best practice: store your credentials in environment variables
wcd_url = os.environ["WCD_URL"]
wcd_api_key = os.environ["WCD_API_KEY"]

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_api_key),             # Replace with your Weaviate Cloud key
    skip_init_checks=True,
)

print(client.is_ready())  # Should print: `True`

def initsystem():
    #login to hugging face
    token = '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'
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

    hugging_face_login(token)
    
    #Using Weaviate for Embedding
    try:
        create_schema()
    except:
        print("Schema already exists")    
    
    all_documents,doccontextrelarr, doccontextutilarr, docadherencearr = get_all_documents_from_ragbench()
    #upload_documents_to_weaviate(all_documents, doccontextrelarr, doccontextutilarr, docadherencearr)
    
initsystem()

#Create Schema of Weaviate
def create_schema():
    try:
      client.collections.create_from_dict({
            'class': 'RAGBenchData_New',
            'vectorizer': 'none',
            'properties': [
                {
                    'name': 'text',
                    'dataType': ['text'],
                    'description': "The text content"
                },
                {
                    'name': 'adherence_score',
                    'dataType': ['number'],
                    'description': "Score indicating adherence"
                },
                {
                    'name': 'context_relevance_score',
                    'dataType': ['number'],
                    'description': "Score indicating relevance of the context"
                },
                {
                    'name': 'context_utilization_score',
                    'dataType': ['number'],
                    'description': "Score indicating how well the context is utilized"
                }
            ]
      })
      print("created schema")
    except Exception as e:
      print(f"Error creating schema: {e}")


# Function to upload documents to Weaviate
def upload_documents_to_weaviate(documents, doccontextrelarr, doccontextutilarr, docadherencearr):
    collection = client.collections.get("RAGBenchData_New")

    index = 0
    for doc_text in documents:
        for doc_sentence in doc_text:
            # Get the embedding for the text
            doc_vector = np.array(embedding_model.encode(doc_sentence), dtype=np.float32).flatten().tolist()
            collection.data.insert(
            {
                'text': doc_sentence,
                'adherence_score': docadherencearr[index],
                'context_relevance_score': doccontextrelarr[index],
                'context_utilization_score': doccontextutilarr[index]
            },
            vector=doc_vector
            )
        index += 1


# Function to retrieve similar documents based on an embedding
def retrieve_similar_documents(query_text, limit=5):
    print(client.is_ready())  # Should print: `True`
    collection = client.collections.get("RAGBenchData_New")

    # Generate embedding for the query text
    query_vector = np.array(embedding_model.encode(query_text), dtype=np.float32).flatten().tolist()  # Convert np.array to list

    # Perform the query to retrieve similar documents
    result = collection.query.near_vector(near_vector=query_vector, limit=limit)

    # Access the results in the expected format
    return result



@app.get("/")
def getValues(query):
    # query
    #query = "What role does T-cell count play in severe human adenovirus type 55 (HAdV-55) infection?"
    #query = "What is Power Saving Mode and Motion Lighting?"
    # Retrieve similar documents
    retrieved_objs = retrieve_similar_documents(query)

    retrieved_docs = []
    print("Retrieved similar documents:")
    for obj in retrieved_objs.objects:
        retrieved_docs.append(obj.properties['text'])
        print(obj)
    
    combined_docs = combine_list_to_string(retrieved_docs)
    apitoken = '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'
    
    generated_response = get_response_attributes(apitoken, query, combined_docs,generated_response )
    print(generated_response)
    return {'message':generated_response}