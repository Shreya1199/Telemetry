#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import pandasql as ps

genai.configure(api_key='AIzaSyBX8VKR1CsVCMprlYT8XmO1W5Wdsz2lOOk')
model = genai.GenerativeModel('gemini-1.0-pro')
os.environ['GOOGLE_API_KEY'] = 'AIzaSyBX8VKR1CsVCMprlYT8XmO1W5Wdsz2lOOk'

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


# In[ ]:


#conversion of embeddings

file_path = "pocMeetingTelemetry.csv"       #replace the file_path for using in different source
df = pd.read_csv(file_path)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
chunk = get_text_chunks(df)
new_db = FAISS.from_texts(chunk, embedding=embeddings)


# In[ ]:


#this is a conversational chain for getting the scenario using marker

def get_conversational_chain_scenario(df):
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

         context:

    {context}
 
    Question:

    {question}
 
    Tasks:

    1. Analyze the user's question to identify the values to filter in the SQL query. 

    2. Construct an SQL query that selects the relevant column (e.g., `SubWorkLoadScenario`) from the table `df` based on the identified column and value.

    3. Always include platform column as mandatory for any outputs with asked columns in any case of the question
 
    Output:

    1. An SQL query that matches the user's request with platform column.
 
    Output Format:

    1.Only the sql query generated in a single line.

    2.Don't include additional information

    3.Don't include any characters above or below the sql squery
 
    Example:

    User Question: "Can you provide me the action where URL is used?"

    SQL Query: SELECT SubWorkLoadScenario, Platform FROM df WHERE action_scenario = 'URL';
 
    """
 
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)
 
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
 
    return chain,context


#this is a function to get response for getting the scenario using marker

def scenario_response(user_question):
    chain,context = get_conversational_chain_scenario(df)
    
    docs = new_db.similarity_search(user_question)
    response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
    
    print(response["output_text"])
    query = response["output_text"]
    output_df = ps.sqldf(query)
    print(output_df)
    print(type(output_df))
    #result = pd.DataFrame(output_df)
    
    return output_df
 


# In[ ]:


#this is the conversational chain for getting the telemetry using scenario
def get_conversational_chain_telemetry(df):
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
     
    1. The data provided is the meeting features telemetry data for Microsoft Teams. Consider yourself a telemetry expert based on the data provided.
     
    2. The data includes four different platforms: iOS, Android, Maglev, and Web. Users have performed the actions mentioned in the "SubWorkLoadScenario" column, and the respective telemetries have been captured.
     
    3. The "SubWorkLoadScenario" column contains the actions performed by the user in Microsoft Teams, and the following columns contain the respective telemetry data:
     
       - "action_gesture"
       - "module_name"
       - "module_type"
       - "action_outcome"
       - "panel_type"
       - "action_scenario"
       - "action_scenario_type"
       - "action_subWorkLoad"
       - "target Thread Type"
       - "thread_type"
       - "databag.rsvp"
       - "module_summary"
       - "target_thread_type"
       - "databag.meeting_rsvp"
       - "databag.is_toggle_on"
       - "databag.community_rsvp"
       - "main_entity_type"
       - "main_slot_app"
       - "eventInfo.identifier"
       - "databag.action_type"
       - "subNav_entity_type"
     
    ### Task: Fetch Telemetries/Markers for a Specific Action
     
    - If the user asks for telemetry/markers for a particular action (e.g., "Give me the telemetry for canceling a meeting"):
      - Search for similar contents in the "SubWorkLoadScenario" column.
      - Identify the platforms where the action is present and fetch those rows.
      - While fetching those rows, provide only the columns that contain values for that specific action from the 21 columns listed above.
      - **Provide only the primary telemetries, found in the "Action Type" column**. **Do not fetch secondary telemetry unless explicitly requested.**
      - **Ignore any columns with empty/null/NaN values.**
     
      **Response Format:**
     
    " The necessary telemetries for canceling a meeting action follows
                       Platform - Andriod
                            Subworkload scenario - Meeting - Cancel a Meeting
                               [ action_gesture - "respective telemetry value"
                                module_name - "respective telemetry value"
                                module_type - "respective telemetry value"
                                action_outcome - "respective telemetry value"
                                panel_type - "respective telemetry value"
                                action_scenario - "respective telemetry value"
                                action_scenario_type - "respective telemetry value"
                                action_subWorkLoad - "respective telemetry value"
                                thread_type - "respective telemetry value"
                                databag.rsvp - "respective telemetry value"
                                module_summary - "respective telemetry value"
                                target_thread_type - "respective telemetry value"
                                databag.meeting_rsvp - "respective telemetry value"
                                databag.is_toggle_on - "respective telemetry value"
                                databag.community_rsvp - "respective telemetry value"
                                main_entity_type - "respective telemetry value"
                                main_slot_app - "respective telemetry value"
                                eventInfo.identifier - "respective telemetry value"
                                databag.action_type - "respective telemetry value"
                                subNav_entity_type - "respective telemetry value" ]
     
     
                        Platform - Maglev
                            Subworkload scenario - Meeting - Cancel a Meeting
                               [ action_gesture - "respective telemetry value"
                                module_name - "respective telemetry value"
                                module_type - "respective telemetry value"
                                action_outcome - "respective telemetry value"
                                panel_type - "respective telemetry value"
                                action_scenario - "respective telemetry value"
                                action_scenario_type - "respective telemetry value"
                                action_subWorkLoad - "respective telemetry value"
                                thread_type - "respective telemetry value"
                                databag.rsvp - "respective telemetry value"
                                module_summary - "respective telemetry value"
                                target_thread_type - "respective telemetry value"
                                databag.meeting_rsvp - "respective telemetry value"
                                databag.is_toggle_on - "respective telemetry value"
                                databag.community_rsvp - "respective telemetry value"
                                main_entity_type - "respective telemetry value"
                                main_slot_app - "respective telemetry value"
                                eventInfo.identifier - "respective telemetry value"
                                databag.action_type - "respective telemetry value"
                                subNav_entity_type - "respective telemetry value" ]
     
                        Platform - Web
                            Subworkload scenario - Meeting - Cancel a Meeting
                               [ action_gesture - "respective telemetry value"
                                module_name - "respective telemetry value"
                                module_type - "respective telemetry value"
                                action_outcome - "respective telemetry value"
                                panel_type - "respective telemetry value"
                                action_scenario - "respective telemetry value"
                                action_scenario_type - "respective telemetry value"
                                action_subWorkLoad - "respective telemetry value"
                                thread_type - "respective telemetry value"
                                databag.rsvp - "respective telemetry value"
                                module_summary - "respective telemetry value"
                                target_thread_type - "respective telemetry value"
                                databag.meeting_rsvp - "respective telemetry value"
                                databag.is_toggle_on - "respective telemetry value"
                                databag.community_rsvp - "respective telemetry value"
                                main_entity_type - "respective telemetry value"
                                main_slot_app - "respective telemetry value"
                                eventInfo.identifier - "respective telemetry value"
                                databag.action_type - "respective telemetry value"
                                subNav_entity_type - "respective telemetry value" ]
     
                        Platform - IOS
                            Subworkload scenario - Meeting - Cancel a Meeting
                               [ action_gesture - "respective telemetry value"
                                module_name - "respective telemetry value"
                                module_type - "respective telemetry value"
                                action_outcome - "respective telemetry value"
                                panel_type - "respective telemetry value"
                                action_scenario - "respective telemetry value"
                                action_scenario_type - "respective telemetry value"
                                action_subWorkLoad - "respective telemetry value"
                                thread_type - "respective telemetry value"
                                databag.rsvp - "respective telemetry value"
                                module_summary - "respective telemetry value"
                                target_thread_type - "respective telemetry value"
                                databag.meeting_rsvp - "respective telemetry value"
                                databag.is_toggle_on - "respective telemetry value"
                                databag.community_rsvp - "respective telemetry value"
                                main_entity_type - "respective telemetry value"
                                main_slot_app - "respective telemetry value"
                                eventInfo.identifier - "respective telemetry value"
                                databag.action_type - "respective telemetry value"
                                subNav_entity_type - "respective telemetry value" ] 
                              "
     
    ### Important Guidelines:
    - **Please provide the output response in bulleted format, and not everything in same line.
    - **Do not provide "Telemetry all" column values unless specifically requested**.
    - **Generate outputs using the provided dataset only**; do not use pre-trained information.
    - **Focus on accuracy**; do not provide extra information.
    """
 
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)
 
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
 
    return chain,context

#this is a function to get response for getting the telemetry using scenario

def telemetry_response(user_question):
    chain,context = get_conversational_chain_telemetry(df)
    
    docs = new_db.similarity_search(user_question)
    response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
    
    result = response["output_text"]
    
    return result



# In[ ]:


#this is the converstational chain for to categorize the prompt

def get_conversational_chain_prompt(user_question):

    prompt_template = """
    context:
    {context}

    Question:
    {question}

    **Instructions**: 
    Understand the user question and category the user prompt or question into either of the below two categories -
        1. Telemetry: If the user prompts for retrieval of telemetry or telemetry markers using any actions or scenarios categorize it into "Telemetry" category
        Example questions - Give me the telemetry markers for join a meeting action 
                          What is the telemetry for creating a community event in Maglev?
        2. Subworkloadscenario: If the user prompts for retrieval of any action using any number of telemetry markers, categorize it into "Subworkloadscenario"
        Example questions - Can you provide me the action where savenewmeeting is used?
                          Can you provide me the action where module type is meetings and panel type is meeting join?
                          Can you provide the action where action acenario is meetingjoin in Web platform?

 
         
    **Output Format**:
        - The output should be either "Telemetry" or "Subworkloadscenario" based on the user question

 
    Example User Query and Expected Response:
    - **User Query**: "Give me the data for joining a meeting through chat button."
    - **Expected Response**: 
        - Telemetry
    - **User Query**: "For what scenario I should use savenewmeeting as Action scenario."
    - **Expected Response**: 
        - Subworkloadscenario  
    - **User Query**: "For what action I should use tap as Action gesture."
    - **Expected Response**: 
        - Subworkloadscenario
    - **User Query**: "FI want the action associated with the following telemetry panel_type - stageSwitcher"
    - **Expected Response**: 
        - Subworkloadscenario       

    """
 
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)
 
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


#this is a function to categorize the prompt

def category_prompt(user_question):

    chain = get_conversational_chain_prompt(df)
    
    response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
    
    if "Telemetry" in response["output_text"]:
        return "Telemetry"
    elif "Subworkloadscenario" in response["output_text"]:
        return "Subworkloadscenario"


# In[ ]:


def user_input2(user_question):
 
    flag = category_prompt(user_question)
    print(flag)
    
    if flag=="Telemetry":
        result = telemetry_response(user_question)
    elif flag=="Subworkloadscenario":
        result = scenario_response(user_question)
        
    return result


 


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
            output = user_input2(user_input)
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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




