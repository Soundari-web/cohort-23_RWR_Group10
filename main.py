from fastapi import FastAPI

#Weaviate related imports
import weaviate
from weaviate.classes.init import Auth
import os
from weaviate.classes.config import Configure
import weaviate.client
from weaviate.classes.init import AdditionalConfig, Timeout, Auth


#SentenceTransformer
from sentence_transformers import SentenceTransformer, util
import numpy as np

from Utilities import * 
import math

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.metrics import mean_squared_error, roc_auc_score
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

app = FastAPI()

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Choose your preferred model

# Load the MT5 model and tokenizer
model_name = 'google/mt5-base'
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Move model to evaluation mode and GPU if available for performance
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#Login to Weaviate
os.environ["WCD_URL"] = "https://ksm5qjrdq3oe5h3nkuc0ag.c0.asia-southeast1.gcp.weaviate.cloud"
os.environ["WCD_API_KEY"] = "3us8OCXm7c9uZ9pHlVb0wXRVIEQbl5Q8vKvH"

# Best practice: store your credentials in environment variables
wcd_url = os.environ["WCD_URL"]
wcd_api_key = os.environ["WCD_API_KEY"]

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_api_key),             # Replace with your Weaviate Cloud key
    skip_init_checks=True,
    additional_config=AdditionalConfig(timeout=Timeout(init=10,insert=3000, query=3000)),
 )


print(client.is_ready())  # Should print: `True`
def chunk_and_embed(text, chunk_size=5, stride=3):
    if isinstance(text, list):
        #print(type(text))
        # Join the list of sentences into a single string if input is a list
        text = " ".join(text)

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)

    embeddings = []

    # Iterate through sentences in sliding window style
    for i in range(0, len(sentences) - chunk_size + 1, stride):
        # Create a chunk of specified size
        chunk = sentences[i:i + chunk_size]
        chunk_text = " ".join(chunk)  # Combine sentences into a single string

        # Create an embedding for the chunk
        chunk_embedding = embedding_model.encode(chunk_text)
        embeddings.append(chunk_embedding)

    return embeddings, sentences


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
            ]
      })
      print("created schema")
    except Exception as e:
      print(f"Error creating schema: {e}")


# Function to upload documents to Weaviate
# Function to upload documents to Weaviate
def upload_documents_to_weaviate(documents):
    collection = client.collections.get("RAGBenchData_New")

    index = 0
    for doc_text in documents:
      # Get the embedding for the text
      chunk_embedding, sentences = chunk_and_embed(doc_text)
      for chunk_vector, sentence in zip(chunk_embedding, sentences):
        doc_vector = np.array(chunk_vector, dtype=np.float32).flatten().tolist()
        # Before sending to Weaviate
        collection.data.insert(
              {
                  'text': sentence,
              },
              vector=doc_vector
          )
        index += 1

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
    
    all_documents = get_all_documents_from_ragbench()
    print(len(all_documents))
 
    #upload_documents_to_weaviate(all_documents)
    
initsystem()


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
def rerank_documents(result, query_text):
    # Prepare the inputs as required for MonoT5
    inputs = []
    for obj in result.objects:
        input_text = f"{query_text} {tokenizer.eos_token} {obj.properties['text']}"
        encoded_input = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        inputs.append(encoded_input)

    scores = []

    # Generate scores for each candidate
    for encoded_input in inputs:
        # Ensure input tensors are moved to the correct device (e.g., GPU if available)
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)

        # Prepare decoder_input_ids. For T5 models, a common approach is to use the <pad> token as decoder input for scoring tasks
        decoder_input_ids = tokenizer.encode(tokenizer.eos_token, return_tensors='pt').to(device)

        with torch.no_grad():
            # For the T5 model, the decoder input is required
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)

            # Use the logits from the last time step for scoring
            score = outputs.logits[:, -1].softmax(dim=-1)
            scores.append(score)

    # Rerank candidates based on scores
    ranked_candidates = sorted(zip(result.objects, scores), key=lambda x: x[1].max().item(), reverse=True)

    document_texts_array = [candidate.properties['text'] for candidate, _ in ranked_candidates]
    print(document_texts_array)

    combined_text = "\\n".join(document_texts_array)
    print (combined_text)
    # Print the ranked results
    for candidate, score in ranked_candidates:
        print(f"Result Object: {candidate}, Score: {score.max().item()}")  # Use .max() to get the highest score

    return combined_text

# Incorporate reranking into your retrieval function
def retrieve_and_rerank(query_text, limit=5):
    initial_result = retrieve_similar_documents(query_text, limit)

    # Now rerank the retrieved documents
    reranked_result = rerank_documents(initial_result, query_text)

    return reranked_result

def checkConnAndConnect():
    if client.is_ready() == False:
        client = weaviate.connect_to_weaviate_cloud(
        cluster_url=wcd_url,                                    # Replace with your Weaviate Cloud URL
        auth_credentials=Auth.api_key(wcd_api_key),             # Replace with your Weaviate Cloud key
        skip_init_checks=True,
    )

@app.get("/")
def getValues(query, datasetname):
    #checkConnAndConnect()
    # query
    #query = "What role does T-cell count play in severe human adenovirus type 55 (HAdV-55) infection?"
    #query = "What is Power Saving Mode and Motion Lighting?"
    # Retrieve similar documents
    combined_docs = retrieve_and_rerank(query)

    apitoken = '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'
    
    response = generate_response(apitoken, query, combined_docs)

    generated_response = get_response_attributes(apitoken, query, combined_docs,response )
    print(generated_response)
    
    #Parse the json
    matches = find_balanced_braces(generated_response)
    for match in matches:
        print("Matched Section:", match)

    input_json_string = matches[0]
    print(input_json_string)
    
    json_data = None
  
    # Attempting to parse the JSON string
    try:
        json_data = json.loads(input_json_string)  # Load JSON string into a Python dictionary
        print("Parsed JSON data:", json_data)
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")
    
    if json_data:
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
        supportkeylen = len(support_keys)
        if supportkeylen != 0:
            completenes = compnum.size / supportkeylen
        else:
            completenes = 0
        print(f"Completeness = {completenes}")

        adherence = 1;

        #Adherence : whether all parts of response are grounded by context
        for val in support_level:
            adherence = val*adherence

            print(f"Adherence = {adherence}")

        
        '''
        #get data from weaviate for the query input
        doccontextrel = 0
        doccontextutil = 0
        docadherence = 0
        for objs in retrieved_objs.objects:
            doccontextrel = obj.properties['context_relevance_score']
            doccontextutil = obj.properties['context_utilization_score']
            docadherence = obj.properties['adherence_score']
            break

        print (doccontextrel)
        print (doccontextutil)
        print (docadherence)
        '''
        doccontextrel, doccontextutil, docadherence =  getdocmetrics(query, datasetname)

        #Compute RMSE, AUCROC

        docadherencearr = np.array([docadherence, 0, 0])
        adherencearr = np.array([adherence, 0, 0])

        #compute RMSE
        rmsecontextrel = mse(doccontextrel, contextrel)
        rmsecontextutil = mse(doccontextutil, contextutil)
        aucscore = -1
        if not math.isnan(docadherence):
            aucscore = roc_auc_score(docadherencearr, adherencearr)

        print(f"RMSE Context Relevance = {rmsecontextrel}")
        print(f"RMSE Context Utilization = {rmsecontextutil}")
        print(f"AUROC Adherence = {aucscore}")
        
        if math.isnan(aucscore):
            aucscore = -1

        return {'message':response, "Context Relevance":contextrel, "Context Utilization": contextutil, "Completeness": completenes, "Adherence": adherence, "RMSE Context Relevance": rmsecontextrel, "RMSE Context Utilization":rmsecontextutil, "AUROC Adherence":aucscore }
    else:
        return{"Error": "JSON data unavailable"}
