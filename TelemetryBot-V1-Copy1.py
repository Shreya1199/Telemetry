#!/usr/bin/env python
# coding: utf-8

# Setup for model

# In[217]:


#!pip install -q -U google-generativeai


# In[11]:


import pathlib
import textwrap

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
import os
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def get_text_from_file(csv_file):
    df = pd.read_csv(csv_file)
    # Convert the entire CSV data to a single text string if desired
    text = df.to_string(index=False)
    return text
 
def get_text_chunks(text):
    chunks = []
    num_rows = len(df)
    chunk_size = 1
    
    for start_row in range(0, num_rows, chunk_size):
        end_row = min(start_row + chunk_size, num_rows)
        chunk = ""
        for index in range(start_row, end_row):
            row = df.iloc[index]
            for col in df.columns:
                chunk += f"{col}: {row[col]}\n"
            chunk += "\n"
        chunks.append(chunk)
    
    return chunks
    
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index2")

def get_conversational_chain(df):
    context = ""
    for index, row in df.iterrows():
        context += f"Row {index+1}:\n"
        for col in df.columns:
            context += f"{col}: {row[col]}\n"
        context += "\n"

    prompt_template = """
    context:
    {context}

    Question:
    {question}
    
    Instructions:
    1. **Telemetry all filtering**:Don't consider/exclude values in "Telemetry all" 
    2. **Platform column filtering**: Filter the rows accordingly based on values "Android","IOS","Web","Maglev" which is relevant to the question
    3.**SubWorkLoadSCenerio column filtering**: Identify the relevant rows where the "SubWorkLoadSCenerio" is relevant to the question
    4. **Additional Column Check**: 
        -If the question contains the word "button", look for the most relevant values in the "SubWorkLoadSCenerio" column that mention "button".
        -If the question doesn't contain the word "button", exclude values in the "SubworkLoadScenario" column that mention "button".
    5. **Null Filtering**: Exclude all columns that have null or nan values.
    6. **Action Type Filtering**: 
        -Identify the relevant rows where the "Action Type" column value is "primary".
        -If in the question mentioned "all telemetries popping up" or "all telemetries", identify and give relevant rows where the "Action Type" column value is both "primary" and "secondary". 
    7. **Output Format**: 
        - Provide a key value pair with column as filtered relevant data
        - Ensure that there are relevant columns and rows based on the instructions above.
        - Exclude columns: Platform, WorkLoad, SubWorkLoad, SubWorkLoadSCenerio, Description, Action Type,Telemetry all.
    
    Example User Query and Expected Response:
    - **User Query**: "Give me the data for joining a meeting through chat button."
    - **Expected Response**: 
        - A key value pair where "Action Type" is "primary" and "SubWorkLoadSCenerio" contains "chat button", excluding null values and it's respective columns.

    """
 
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)
 
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
 
    return chain,context
 
def user_input(user_question):
 
    chain,context = get_conversational_chain(df)

    docs = new_db.similarity_search(user_question)
    response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
 
    return response["output_text"]

def firstfunction(user_question):
    
    answer = user_input(user_question)
    return answer
    # answer_df = pd.DataFrame(answer)
    #print("Reply:\n\n", answer)

#Converting Top Operator to Limit Operator as pandasql doesn't support Top
def convert_top_to_limit(sql):
    try:
        tokens = sql.upper().split()
        is_top_used = False

        for i, token in enumerate(tokens):
            if token == 'TOP':
                is_top_used = True
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    limit_value = tokens[i + 1]
                    # Remove TOP and insert LIMIT and value at the end
                    del tokens[i:i + 2]
                    tokens.insert(len(tokens), 'LIMIT')
                    tokens.insert(len(tokens), limit_value)
                    break  # Exit loop after successful conversion
                else:
                    raise ValueError("TOP operator should be followed by a number")

        return ' '.join(tokens) if is_top_used else sql
    except Exception as e:
        err = f"An error occurred while converting Top to Limit in SQL Query: {e}"
        return err

#Function to add Table Name into the SQL Query as it is, as the Table Name is Case Sensitive here
def process_tablename(sql, table_name):
    try:
        x = sql.upper()
        query = x.replace(table_name.upper(), table_name)
        return query
    except Exception as e:
        err = f"An error occurred while processing table name in SQL query: {e}"
        return err

    
    
genai.configure(api_key='AIzaSyAsuwwlZy0FNQpYzzWrZNoNnF2w5edkTlI')
model = genai.GenerativeModel('gemini-1.0-pro')
response = model.generate_content("what is apple as a fruit")
# print(response.text)

#!pip install streamlit
#!pip install PyPDF2
#!pip install python-dotenv
#!pip install langchain
#!pip install faiss-cpu
#!pip install langchain_google_genai
#%env GOOGLE_API_KEY=AIzaSyAsuwwlZy0FNQpYzzWrZNoNnF2w5edkTlI



os.environ['GOOGLE_API_KEY'] = 'AIzaSyAsuwwlZy0FNQpYzzWrZNoNnF2w5edkTlI'

#!pip install -U langchain-community

#conversion of embeddings
file_path = "pocMeetingTelemetry.xlsx"
df = pd.read_excel(file_path)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
chunk = get_text_chunks(df)
new_db = FAISS.from_texts(chunk, embedding=embeddings)


# print(chunk)





quantdf  = pd.read_excel("pocMeetingTelemetry.xlsx")


# In[10]:




# In[ ]:


def main():
    try:

        # Create a container for logos and title with horizontal layout
        col1, col2, col3 = st.columns([1, 2, 1])
      
        # Display logo on the left
        with col1:
            st.image("musigmalogo3.jfif", width=50)  # Adjust width as needed

        # Display title in the center
        with col2:
            st.header("Telemetry Bot")

        # Display logo on the right
        with col3:
            st.image("msftlogo4.jpg", width=50)  # Align the logo to the right
      
        # User input section
        user_input = st.text_input("Please enter the question:", placeholder="What would you like to process?")

        # Process button and output section
        if st.button("Process"):
            output = firstfunction(user_input)
            #t.session_state['chat_history'].append((user_input, output))
        
            # Display output based on type (string or dataframe)
            if isinstance(output, pd.DataFrame):
                st.dataframe(output)
            else:
                st.write(output)

        # Chat history section with some formatting
#         st.header("Chat History")
#         for user_text, output_text in st.session_state['chat_history']:
#             st.markdown(f"- You: {user_text}")
#             if isinstance(output_text, pd.DataFrame):
#                 st.dataframe(output_text)  # Convert dataframe to string for display
#             else:
#                 st.markdown(f"- Bot: {output_text}")
#             st.write("---")
    except Exception as e:
        err = f"An error occurred while calling the final function: {e}"
        return err


if __name__ == "__main__":
    main()




