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
from sklearn.metrics import mean_squared_error, roc_auc_score


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
    
    generated_response = generate_response(apitoken, query, combined_docs)

    generated_response = get_response_attributes(apitoken, query, combined_docs,generated_response )
    print(generated_response)
    
    #Parse the json
    matches = find_balanced_braces(generated_response)
    for match in matches:
        print("Matched Section:", match)

    input_json_string = matches[0]
    print(input_json_string)
    
    
    # Attempting to parse the JSON string
    try:
        json_data = json.loads(input_json_string)  # Load JSON string into a Python dictionary
        print("Parsed JSON data:", json_data)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
    
    relavance_explanation = json_data['relevance_explanation']
    relevant_sentence_keys = json_data['all_relevant_sentence_keys']
    overall_supported_explanation = json_data['overall_supported_explanation']
    overall_supported = json_data['overall_supported']
    sentence_support_information = json_data['sentence_support_information']
    all_utilized_sentence_keys = json_data['all_utilized_sentence_keys']

    print (sentence_support_information)
    support_keys = []
    support_level = []
    for sentence_support in sentence_support_information:
        support_keys += sentence_support['supporting_sentence_keys']
        support_level.append(sentence_support['fully_supported'])

    #compute Context Relevance
    contextrel = compute_context_relevance(relevant_sentence_keys, support_keys)
    print(f"Context Relevance = {contextrel}")

    contextutil = compute_context_utilization(relevant_sentence_keys, all_utilized_sentence_keys)
    print(f"Context Utilization = {contextutil}")

    compnum = np.intersect1d(support_keys, all_utilized_sentence_keys)
    completenes = compnum.size / len(support_keys)
    print(f"Completeness = {completenes}")

    adherence = 1;

    #Adherence : whether all parts of response are grounded by context
    for val in support_level:
        adherence = val*adherence

        print(f"Adherence = {adherence}")

    #get data from weaviate for the query input
    doccontextrel = 0
    doccontextutil = 0
    docadherence = 0
    for objs in retrieved_objs.objects:
        doccontextrel = obj.properties['context_relevance_score']
        doccontextutil = obj.properties['context_utilization_score']
        doccontextutil = obj.properties['adherence_score']
        break

    print (doccontextrel)
    print (doccontextutil)
    print (docadherence)

    #Compute RMSE, AUCROC

    docadherencearr = np.array([docadherence, 0, 0])
    adherencearr = np.array([adherence, 0, 0])

    #compute RMSE
    rmsecontextrel = mse(doccontextrel, contextrel)
    rmsecontextutil = mse(doccontextutil, contextutil)
    aucscore = roc_auc_score(docadherencearr, adherencearr)

    print(f"RMSE Context Relevance = {rmsecontextrel}")
    print(f"RMSE Context Utilization = {rmsecontextutil}")
    print(f"AUROC Adherence = {aucscore}")


    return {'message':generated_response, "Context Relevance":contextrel, "Context Utilization": contextutil, "Completeness": completenes, "Adherence": adherence, "RMSE Context Relevance": rmsecontextrel, "RMSE Context Utilization":rmsecontextutil}