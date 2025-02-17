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
    doccontextrelarr = covidqadataset['train']['relevance_score']
    doccontextutilarr = covidqadataset['train']['utilization_score']
    docadherencearr = list(map(float, covidqadataset['train']['adherence_score']))
    doccontextrelarr += cuaddataset['train']['relevance_score']
    doccontextutilarr += cuaddataset['train']['utilization_score']
    docadherencearr += list(map(float, cuaddataset['train']['adherence_score']))
    doccontextrelarr += delucionqaadataset['train']['relevance_score']
    doccontextutilarr += delucionqaadataset['train']['utilization_score']
    docadherencearr += [float(x) if x is not None else 0.0 for x in delucionqaadataset['train']['adherence_score']]
    doccontextrelarr += emanualdataset['train']['relevance_score']
    doccontextutilarr += emanualdataset['train']['utilization_score']
    docadherencearr += list(map(float, emanualdataset['train']['adherence_score']))
    doccontextrelarr += expertqadataset['train']['relevance_score']
    doccontextutilarr += expertqadataset['train']['utilization_score']
    docadherencearr += list(map(float, expertqadataset['train']['adherence_score']))
    doccontextrelarr += finqadataset['train']['relevance_score']
    doccontextutilarr += finqadataset['train']['utilization_score']
    docadherencearr += list(map(float, finqadataset['train']['adherence_score']))
    doccontextrelarr += hagriddataset['train']['relevance_score']
    doccontextutilarr += hagriddataset['train']['utilization_score']
    docadherencearr += list(map(float, hagriddataset['train']['adherence_score']))
    doccontextrelarr += hotpotqadataset['train']['relevance_score']
    doccontextutilarr += hotpotqadataset['train']['utilization_score']
    docadherencearr += list(map(float, hotpotqadataset['train']['adherence_score']))
    doccontextrelarr += msmarcodataset['train']['relevance_score']
    doccontextutilarr += msmarcodataset['train']['utilization_score']
    docadherencearr += list(map(float, msmarcodataset['train']['adherence_score']))
    doccontextrelarr += pubmedqadataset['train']['relevance_score']
    doccontextutilarr += pubmedqadataset['train']['utilization_score']
    docadherencearr += list(map(float, pubmedqadataset['train']['adherence_score']))
    doccontextrelarr += tatqadataset['train']['relevance_score']
    doccontextutilarr += tatqadataset['train']['utilization_score']
    docadherencearr += list(map(float, tatqadataset['train']['adherence_score']))
    doccontextrelarr += techqadataset['train']['relevance_score']
    doccontextutilarr += techqadataset['train']['utilization_score']
    docadherencearr += list(map(float, techqadataset['train']['adherence_score']))
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
    I asked someone to answer a question based on one or more documents.
    Your task is to review their response and assess whether or not each sentence
    in that response is supported by text in the documents. And if so, which
    sentences in the documents provide that support. You will also tell me which
    of the documents contain useful information for answering the question, and
    which of the documents the answer was sourced from.
    Here are the documents, each of which is split into sentences. Alongside each
    sentence is associated key, such as ’0a.’ or ’0b.’ that you can use to refer
    to it:
    ‘‘‘
    {documents}
    ‘‘‘
    The question was:
    ‘‘‘
    {question}
    ‘‘‘
    Here is their response, split into sentences. Alongside each sentence is
    associated key, such as ’a.’ or ’b.’ that you can use to refer to it. Note
    that these keys are unique to the response, and are not related to the keys
    in the documents:
    ‘‘‘
    {answer}
    ‘‘‘
    You must respond with a JSON object matching this schema:
    ‘‘‘ {{
      "relevance_explanation": string,
      "all_relevant_sentence_keys": [string],
      "overall_supported_explanation": string,
      "overall_supported": boolean,
      "sentence_support_information": [
        {{
          "response_sentence_key": string,
          "explanation": string,
    16
          "supporting_sentence_keys": [string],
          "fully_supported": boolean
        }},
    ],
      "all_utilized_sentence_keys": [string]
    }}
    ‘‘‘
    The relevance_explanation field is a string explaining which documents
    contain useful information for answering the question. Provide a step-by-step
    breakdown of information provided in the documents and how it is useful for
    answering the question.
    The all_relevant_sentence_keys field is a list of all document sentences keys
    (e.g. ’0a’) that are revant to the question. Include every sentence that is
    useful and relevant to the question, even if it was not used in the response,
    or if only parts of the sentence are useful. Ignore the provided response when
    making this judgement and base your judgement solely on the provided documents
    and question. Omit sentences that, if removed from the document, would not
    impact someone’s ability to answer the question.
    The overall_supported_explanation field is a string explaining why the response
    *as a whole* is or is not supported by the documents. In this field, provide a
    step-by-step breakdown of the claims made in the response and the support (or
    lack thereof) for those claims in the documents. Begin by assessing each claim
    separately, one by one; don’t make any remarks about the response as a whole
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
    - supporting_sentence_keys: keys (e.g. ’0a’) of sentences from the documents that
    support the response sentence. If the sentence is not supported, this list MUST
    be empty. If the sentence is supported, this list MUST contain one or more keys.
    In special cases where the sentence is supported, but not by any specific sentence,
    you can use the string "supported_without_sentence" to indicate that the sentence
    is generally supported by the documents. Consider cases where the sentence is
    expressing inability to answer the question due to lack of relevant information in
    the provided contex as "supported_without_sentence". In cases where the sentence
    is making a general statement (e.g. outlining the steps to produce an answer, or
    summarizing previously stated sentences, or a transition sentence), use the
    sting "general".In cases where the sentence is correctly stating a well-known fact,
    like a mathematical formula, use the string "well_known_fact". In cases where the
    sentence is performing numerical reasoning (e.g. addition, multiplication), use
    the string "numerical_reasoning".
    - fully_supported: a boolean indicating whether the sentence is fully supported by
    the documents.
      - This value should reflect the conclusion you drew at the end of your step-by-step
      breakdown in explanation.
      - If supporting_sentence_keys is an empty list, then fully_supported must be false.
    17

    - Otherwise, use fully_supported to clarify whether everything in the response
      sentence is fully supported by the document text indicated in supporting_sentence_keys
      (fully_supported = true), or whether the sentence is only partially or incompletely
      supported by that document text (fully_supported = false).
    The all_utilized_sentence_keys field is a list of all sentences keys (e.g. ’0a’) that
    were used to construct the answer. Include every sentence that either directly supported
    the answer, or was implicitly used to construct the answer, even if it was not used
    in its entirety. Omit sentences that were not used, and could have been removed from
    the documents without affecting the answer.
    You must respond with a valid JSON string.  Use escapes for quotes, e.g. ‘\\"‘, and
    newlines, e.g. ‘\\n‘. Do not write anything before or after the JSON string. Do not
    wrap the JSON string in backticks like ‘‘‘ or ‘‘‘json.
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

    for sentence in relevant_sentences:
      if sentence in support_keys:
        total_relevance_score += 1

    # To avoid division by zero in case there are no relevant sentences
    if total_relevant_sentences == 0:
        return 0

    return total_relevance_score / total_relevant_sentences

def compute_context_utilization(relevant_sentences, utilization_levels):
    total_utilization_score = 0
    total_relevant_sentences = len(relevant_sentences)
    for sentence in relevant_sentences:
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




def mse(actual, predicted):
    if type(actual) == float and type(predicted) == float:
        return (actual - predicted)**2
    else:
        return -1
  
#get document context rel, utilization, adherence from the specified dataset for a specific query

def getdocmetrics(query, datasetname):

  doccontextrel = 0
  doccontextutil = 0
  docadherence = 0
  ind = 0
  dataset = load_dataset("rungalileo/ragbench", datasetname)
  for question in dataset['train']['question']:
  
    if question == query:
      print('found')
      doccontextrel = dataset['train'][ind]['relevance_score']
      doccontextutil = dataset['train'][ind]['utilization_score']
      docadherence = float(dataset['train'][ind]['adherence_score'])
      break
    else:
      doccontextrel= -1
      doccontextutil = -1
      docadherence = -1
    ind += 1


  print (doccontextrel)
  print (doccontextutil)
  print (docadherence)
  return doccontextrel, doccontextutil, docadherence

