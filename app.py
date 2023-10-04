import openai
import gradio as gr
import os
import time
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import Vector  
from azure.search.documents.indexes.models import (  
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    PrioritizedFields,  
    SemanticField,  
    SearchField,  
    SemanticSettings,  
    VectorSearch,  
    HnswVectorSearchAlgorithmConfiguration,  
)  

from dotenv import dotenv_values
# specify the name of the .env file name 
env_name = "env.env" # change to use your own .env file
config = dotenv_values(env_name)

#openai.api_key = "sk-..."  # Replace with your key
openai.api_type = "azure"
openai.api_key = config["AZURE_OPENAI_API_KEY"]
openai.api_base = config["AZURE_OPENAI_ENDPOINT"]
openai.api_version = config["AZURE_OPENAI_API_VERSION"]

model = config["AZURE_OPENAI_MODEL"]
deployment = config["AZURE_OPENAI_DEPLOYMENT"]

def getmebedding(text):
    response = openai.Embedding.create(
    input=text,
    engine="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def search(query):
    service_endpoint = config["AZURE_SEARCH_ENDPOINT"]
    key = config["AZURE_SEARCH_API_KEY"]
    index_name = config["AZURE_SEARCH_INDEX"]
    resultstr = ""
    search_client = SearchClient(service_endpoint, index_name, AzureKeyCredential(key))
    vector = Vector(value=getmebedding(query), k=3, fields="chunkVector")  

    results = search_client.search(  
        search_text=query,  
        vectors=[vector],
        select=["title", "chunk", "name"],
        query_type="semantic", query_language="en-us", semantic_configuration_name='my-semantic-config', query_caption="extractive", query_answer="extractive",
        top=3
    )

    semantic_answers = results.get_answers()
    for answer in semantic_answers:
        if answer.highlights:
            print(f"Semantic Answer: {answer.highlights}")
        else:
            print(f"Semantic Answer: {answer.text}")
        print(f"Semantic Answer Score: {answer.score}\n")

    for result in results:
        #print(f"Title: {result['title']}")
        #print(f"Content: {result['chunk']}")
        #print(f"Category: {result['name']}")
        #result.items(str(result['chunk']))
        resultstr = resultstr + result['chunk']

        captions = result["@search.captions"]
        if captions:
            caption = captions[0]
            #if caption.highlights:
            #    print(f"Caption: {caption.highlights}\n")
            #else:
            #    print(f"Caption: {caption.text}\n")
    return resultstr

def predict(message, history):
    history_openai_format = []
    content = search(message)

    messagecontent = f"""Assistant helps the company HR employees with their employee skills for job position, and questions about the employee profiles. Be brief in your answers.
    Answer ONLY with the facts listed in the list of sources below. If there isn't enough information below, say you don't know. Do not generate answers that don't use the sources below.
    If asking a clarifying question to the user would help, ask the question.
    For tabular information return it as an html table. Do not return markdown format. If the question is not in English, answer in the language used in the question.
    Each source has a name followed by colon and the actual information, always include the source name for each fact you use in the response. 
    Use square brackets to reference the source, e.g. [info1.txt]. Don't combine sources, list each source separately, e.g. [info1.txt][info2.pdf].
    

    Sources: 
    {content}

    Question:
    {message}    
    """

    messagecontent = f"""{message} \n\nSources:\n{content} can you provide citations as links to the sources?"""
    
    #print(messagecontent)
    print(content)

    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    #history_openai_format.append({"role": "user", "content": messagecontent})
    history_openai_format.append({"role": "user", "content": message + "\n\nSources:\n" + content})
    

    #print('Search: ' + str(search(message)))

    functions = [
            {
                "name": "search_sources",
                "description": "Retrieve sources from the Azure Cognitive Search index",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Query string to retrieve documents from azure search eg: 'Health care plan'",
                        }
                    },
                    "required": ["search_query"],
                },
            }
        ]

    # record the time before the request is sent
    start_time = time.time()
    response = openai.ChatCompletion.create(
        model=model,
        deployment_id=deployment,
        messages= history_openai_format,
        temperature=1.0,
        stream=True
    )

    collected_chunks = []
    collected_messages = []

    partial_message = ""
    for chunk in response:
        #print('chunk ' + str(chunk))
        if(len(chunk['choices']) > 0):
            if len(chunk['choices'][0]['delta']) > 0:
                #print('delta ' + str(chunk['choices'][0]['delta']))
                if 'content' in chunk['choices'][0]['delta']:
                    #print('delta ' + str(chunk['choices'][0]['delta']))
                    partial_message = partial_message + str(chunk['choices'][0]['delta']['content'])
                    yield partial_message


gr.ChatInterface(predict, chatbot=gr.Chatbot(height=600),title="Profile Chat Bot", description="Ask me any question", theme="soft", clear_btn="Clear",).queue().launch()