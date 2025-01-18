from huggingface_hub import notebook_login
from huggingface_hub import HfApi
from huggingface_hub import HfFolder
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np
from transformers import pipeline
#from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch
import os
from langchain_huggingface import HuggingFacePipeline
from langchain.llms import HuggingFaceEndpoint

from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate


from datasets import load_dataset
from huggingface_hub import login
import re
import json

#login to hugging face
def hugging_face_login(token):
    # Login to Hugging Face
    login(token)
    print("Logged in to Hugging Face successfully.")

#getting all documents from ragbench dataset
def get_all_documents_from_ragbench():
  
    # Load the RAGBench dataset from Hugging Face
    covidqadataset = load_dataset("rungalileo/ragbench", 'covidqa')
    cuaddataset = load_dataset("rungalileo/ragbench", 'cuad')
    delucionqaadataset = load_dataset("rungalileo/ragbench", 'delucionqa')
    emanualdataset = load_dataset("rungalileo/ragbench", 'emanual')
    expertqadataset = load_dataset("rungalileo/ragbench", 'expertqa')
    finqadataset = load_dataset("rungalileo/ragbench", 'finqa')
    hagriddataset = load_dataset("rungalileo/ragbench", 'hagrid')
    hotpotqadataset = load_dataset("rungalileo/ragbench", 'hotpotqa')
    msmarcodataset = load_dataset("rungalileo/ragbench", 'msmarco')
    pubmedqadataset = load_dataset("rungalileo/ragbench", 'pubmedqa')
    tatqadataset = load_dataset("rungalileo/ragbench", 'tatqa')
    techqadataset = load_dataset("rungalileo/ragbench", 'techqa')
    # Assuming 'train' split contains the documents, modify if you need a different split
    documents = covidqadataset['train']['documents']
    documents += cuaddataset['train']['documents']
    documents += delucionqaadataset['train']['documents']
    documents += emanualdataset['train']['documents']
    documents += expertqadataset['train']['documents']
    documents += finqadataset['train']['documents']
    documents += hagriddataset['train']['documents']
    documents += hotpotqadataset['train']['documents']
    documents += msmarcodataset['train']['documents']
    documents += pubmedqadataset['train']['documents']
    documents += tatqadataset['train']['documents']
    documents += techqadataset['train']['documents']
    doccontextrelarr = covidqadataset['train']['gpt3_context_relevance']
    doccontextutilarr = covidqadataset['train']['gpt35_utilization']
    docadherencearr = covidqadataset['train']['gpt3_adherence']
    doccontextrelarr += cuaddataset['train']['gpt3_context_relevance']
    doccontextutilarr += cuaddataset['train']['gpt35_utilization']
    docadherencearr += cuaddataset['train']['gpt3_adherence']
    doccontextrelarr += delucionqaadataset['train']['gpt3_context_relevance']
    doccontextutilarr += delucionqaadataset['train']['gpt35_utilization']
    docadherencearr += delucionqaadataset['train']['gpt3_adherence']
    doccontextrelarr += emanualdataset['train']['gpt3_context_relevance']
    doccontextutilarr += emanualdataset['train']['gpt35_utilization']
    docadherencearr += emanualdataset['train']['gpt3_adherence']
    doccontextrelarr += expertqadataset['train']['gpt3_context_relevance']
    doccontextutilarr += expertqadataset['train']['gpt35_utilization']
    docadherencearr += expertqadataset['train']['gpt3_adherence']
    doccontextrelarr += finqadataset['train']['gpt3_context_relevance']
    doccontextutilarr += finqadataset['train']['gpt35_utilization']
    docadherencearr += finqadataset['train']['gpt3_adherence']
    doccontextrelarr += hagriddataset['train']['gpt3_context_relevance']
    doccontextutilarr += hagriddataset['train']['gpt35_utilization']
    docadherencearr += hagriddataset['train']['gpt3_adherence']
    doccontextrelarr += hotpotqadataset['train']['gpt3_context_relevance']
    doccontextutilarr += hotpotqadataset['train']['gpt35_utilization']
    docadherencearr += hotpotqadataset['train']['gpt3_adherence']
    doccontextrelarr += msmarcodataset['train']['gpt3_context_relevance']
    doccontextutilarr += msmarcodataset['train']['gpt35_utilization']
    docadherencearr += msmarcodataset['train']['gpt3_adherence']
    doccontextrelarr += pubmedqadataset['train']['gpt3_context_relevance']
    doccontextutilarr += pubmedqadataset['train']['gpt35_utilization']
    docadherencearr += pubmedqadataset['train']['gpt3_adherence']
    doccontextrelarr += tatqadataset['train']['gpt3_context_relevance']
    doccontextutilarr += tatqadataset['train']['gpt35_utilization']
    docadherencearr += tatqadataset['train']['gpt3_adherence']
    doccontextrelarr += techqadataset['train']['gpt3_context_relevance']
    doccontextutilarr += techqadataset['train']['gpt35_utilization']
    docadherencearr += techqadataset['train']['gpt3_adherence']
    return documents, doccontextrelarr, doccontextutilarr, docadherencearr
  
  #generating response - using HuggingFace Endpoint
def generate_response(apitoken, question, documents, max_length=500):
  repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

  llm = HuggingFaceEndpoint(
      repo_id=repo_id,
      max_length=128,
      temperature=0.5,
      huggingfacehub_api_token=apitoken,
  )
  template = f"Using the following documents, answer the question.\\nDocuments:\\n{documents}\\n\\nQuestion: {question}\\nAnswer:"
  prompt = PromptTemplate.from_template(template)
  llm_chain = prompt | llm
  print(prompt)
  response = llm_chain.invoke({"question": question})
  return response

  
#combine the retrieved documents to a string
def combine_list_to_string(text_list, separator="\\n"):
    combined_string = ""
    for text in text_list:
      combined_string += str(text)
      combined_string += separator
    return combined_string

#combined_docs = combine_list_to_string(retrieved_docs)
apitoken = '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'
#generated_response = generate_response(apitoken, query, combined_docs)
#generated_response = answer_question(combined_docs, query)
#print(f"Generated Response: {generated_response}")

#getting query response attributes by using a prompt
def get_response_attributes(apitoken, question, documents, answer, max_length=500):

  repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
  model_kwargs = {
    "max_length": 2048,
  }

  llm = HuggingFaceEndpoint(
      repo_id=repo_id,
      temperature=0.5,
      max_new_tokens=2048,
      huggingfacehub_api_token=apitoken,
      model_kwargs=model_kwargs,
  )

  template = """
      You asked someone to answer a question based on one or more documents.
      Your task is to review their response and assess whether or not each sentence
      in that response is supported by text in the documents. And if so, which
      sentences in the documents provide that support. You will also tell me which
      of the documents contain useful information for answering the question, and
      which of the documents the answer was sourced from.
      Here are the documents, each of which is split into sentences. Alongside each
      sentence is an associated key, such as '0a.' or '0b.' that you can use to refer
      to it:
      ‘’’
      {documents}
      ‘’’
      The question was:
      ‘’’
      {question}
      ‘’’
      Here is their response, split into sentences. Alongside each sentence is
      an associated key, such as 'a.' or 'b.' that you can use to refer to it. Note
      that these keys are unique to the response, and are not related to the keys
      in the documents:
      ‘’’
      {answer}
      ‘’’
      You must respond with a JSON object matching this schema:
      ‘’’
      {{
      "relevance_explanation": string,
      "all_relevant_sentence_keys": [string],
      "overall_supported_explanation": string,
      "overall_supported": boolean,
      "sentence_support_information": [
      {{
      "response_sentence_key": string,
      "explanation": string,
      "supporting_sentence_keys": [string],
      "fully_supported": boolean
      }},
      ],
      "all_utilized_sentence_keys": [string]
      }}
      ‘’’
      The relevance_explanation field is a string explaining which documents
      contain useful information for answering the question. Provide a step-by-step
      breakdown of information provided in the documents and how it is useful for
      answering the question.
      The all_relevant_sentence_keys field is a list of all document sentences keys
      (e.g. '0a') that are relevant to the question. Include every sentence that is
      useful and relevant to the question, even if it was not used in the response,
      or if only parts of the sentence are useful. Ignore the provided response when
      making this judgment and base your judgment solely on the provided documents
      and question. Omit sentences that, if removed from the document, would not
      impact someone's ability to answer the question.
      The overall_supported_explanation field is a string explaining why the response
      *as a whole* is or is not supported by the documents. In this field, provide a
      step-by-step breakdown of the claims made in the response and the support (or
      lack thereof) for those claims in the documents. Begin by assessing each claim
      separately, one by one; don't make any remarks about the response as a whole
      until you have assessed all the claims in isolation.
      The overall_supported field is a boolean indicating whether the response as a
      whole is supported by the documents. This value should reflect the conclusion
      you drew at the end of your step-by-step breakdown in overall_supported_explanation.
      In the sentence_support_information field, provide information about the support
      *for each sentence* in the response.
      The sentence_support_information field is a list of objects, one for each sentence
      in the response. Each object MUST have the following fields:
      - response_sentence_key: a string identifying the sentence in the response.
      This key is the same as the one used in the response above.
      - explanation: a string explaining why the sentence is or is not supported by the
      documents.
      - supporting_sentence_keys: keys (e.g. '0a') of sentences from the documents that
      support the response sentence. If the sentence is not supported, this list MUST
      be empty. If the sentence is supported, this list MUST contain one or more keys.
      In special cases where the sentence is supported, but not by any specific sentence,
      you can use the string "supported_without_sentence" to indicate that the sentence
      is generally supported by the documents. Consider cases where the sentence is
      expressing inability to answer the question due to lack of relevant information in
      the provided context as "supported_without_sentence". In cases where the
      sentence is making a general statement (e.g. outlining the steps to produce an answer, or
      summarizing previously stated sentences, or a transition sentence), use the
      string "general". In cases where the sentence is correctly stating a well-known fact,
      like a mathematical formula, use the string "well_known_fact". In cases where the
      sentence is performing numerical reasoning (e.g. addition, multiplication), use the
      string "numerical_reasoning".
      - fully_supported: a boolean indicating whether the sentence is fully supported by
      the documents.
      - This value should reflect the conclusion you drew at the end of your step-by-step
      breakdown in explanation.
      - If supporting_sentence_keys is an empty list, then fully_supported must be false.
      - Otherwise, use fully_supported to clarify whether everything in the response
      sentence is fully supported by the document text indicated in supporting_sentence_keys
      (fully_supported = true), or whether the sentence is only partially or incompletely
      supported by that document text (fully_supported = false).
      The all_utilized_sentence_keys field is a list of all sentences keys (e.g. '0a') that
      were used to construct the answer. Include every sentence that either directly supported
      the answer, or was implicitly used to construct the answer, even if it was not used
      in its entirety. Omit sentences that were not used and could have been removed from
      the documents without affecting the answer.
      You must respond with a valid JSON string. Use escapes for quotes, e.g. '\\\\"', and
      newlines, e.g. '\\\\n'. Do not write anything before or after the JSON string. Do not
      wrap the JSON string in backticks like '\\`' or '\\`json.
      As a reminder: your task is to review the response and assess which documents contain
      useful information pertaining to the question, and how each sentence in the response
      is supported by the text in the documents.
      """

  prompt = PromptTemplate.from_template(template)
  llm_chain = prompt | llm
  response = llm_chain.invoke({"question": question, "documents": documents, "answer": answer})

  return response


#Defined as utilized documents / retrieved documents for the query
def compute_context_relevance(relevant_sentences, support_keys):
    total_relevance_score = 0
    total_relevant_sentences = len(relevant_sentences)

    for sentence in relevant_sentence_keys:
      if sentence in support_keys:
        total_relevance_score += 1

    # To avoid division by zero in case there are no relevant sentences
    if total_relevant_sentences == 0:
        return 0

    return total_relevance_score / total_relevant_sentences

def compute_context_utilization(relevant_sentences, utilization_levels):
    total_utilization_score = 0
    total_relevant_sentences = len(relevant_sentences)
    for sentence in relevant_sentence_keys:
      if sentence in utilization_levels:
        total_utilization_score += 1
    # To avoid division by zero in case there are no relevant sentences
    if total_relevant_sentences == 0:
        return 0
    return total_utilization_score / total_relevant_sentences
  
#parse the response for JSON
def find_balanced_braces(input_string):
    stack = []
    matched_sections = []
    current_start = None

    for index, char in enumerate(input_string):
        if char == '{':
            if not stack:
                # Mark the start of a new section
                current_start = index
            stack.append(char)  # Push the opening brace onto the stack
        elif char == '}':
            if stack:
                stack.pop()  # Pop the last opening brace
                if not stack:
                    # If stack is empty, we found a balanced section
                    matched_sections.append(input_string[current_start:index + 1])
                    current_start = None  # Reset for the next match

    return matched_sections
'''
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

#print(support_keys)
#print(support_level)
#compute Context Relevance
contextrel = compute_context_relevance(relevant_sentence_keys, support_keys)
print(f"Context Relevance = {contextrel}")

contextutil = compute_context_utilization(relevant_sentence_keys, all_utilized_sentence_keys)
print(f"Context Utilization = {contextutil}")

compnum = np.intersect1d(support_keys, all_utilized_sentence_keys)
completenes = compnum.size / len(support_keys)
print(f"Completeness = {completenes}")


#Adherence : whether all parts of response are grounded by context
for val in support_level:
  adherence = 1;
  adherence = val*adherence

print(f"Adherence = {adherence}")

def mse(actual, predicted):
    return (actual - predicted)**2
  

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
from sklearn.metrics import mean_squared_error, roc_auc_score

docadherencearr = np.array([docadherence, 0, 0])
adherencearr = np.array([adherence, 0, 0])

#compute RMSE
rmsecontextrel = mse(doccontextrel, contextrel)
rmsecontextutil = mse(doccontextutil, contextutil)
aucscore = roc_auc_score(docadherencearr, adherencearr)

print(f"RMSE Context Relevance = {rmsecontextrel}")
print(f"RMSE Context Utilization = {rmsecontextutil}")
print(f"AUROC Adherence = {aucscore}")


#Compute RMSE, AUCROC
from sklearn.metrics import mean_squared_error, roc_auc_score

doccontextrel = 0
doccontextutil = 0
docadherence = 0
ind = 0
#get gpt3_context_relevance, gpt35_utilization, gpt3_adherence from documents
for question in covidqadataset['train']['question']:
#for question in emanualdataset['train']['question']:
  if question == query:
    print('found')
    doccontextrel = covidqadataset['train'][ind]['gpt3_context_relevance']
    doccontextutil = covidqadataset['train'][ind]['gpt35_utilization']
    docadherence = covidqadataset['train'][ind]['gpt3_adherence']
    break
  else:
    doccontextrel= -1
    doccontextutil = -1
    docadherence = -1
  ind += 1


print (doccontextrel)
print (doccontextutil)
print (docadherence)

docadherencearr = np.array([docadherence, 0, 0])
adherencearr = np.array([adherence, 0, 0])

print (docadherencearr)



#compute RMSE
rmsecontextrel = mse(doccontextrel, contextrel)
rmsecontextutil = mse(doccontextutil, contextutil)
aucscore = roc_auc_score(docadherencearr, adherencearr)

print(f"RMSE Context Relevance = {rmsecontextrel}")
print(f"RMSE Context Utilization = {rmsecontextutil}")
print(f"AUROC Adherence = {aucscore}")

'''
