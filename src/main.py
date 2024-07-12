# Import necessary modules and set up the OpenAI API key
import os
from apikey import openai_api_key

import streamlit as st
import pandas as pd
from langchain_mistralai import ChatMistralAI
from langchain.llms import OpenAI
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
# from langchain.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool 
from langchain.agents.agent_types import AgentType
# from langchain.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv, find_dotenv
from langchain_mistralai import ChatMistralAI
from openai import RateLimitError
from langchain_core.messages import HumanMessage



import os


# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
# llm0 = ChatMistralAI(model="mistral-large-latest")

from langchain_experimental.agents import create_pandas_dataframe_agent
# Set the OpenAI API key from the environment variable
os.environ['OPENAI_API_KEY'] = openai_api_key






# llm0 = ChatMistralAI(model="mistral-large-latest")
# Load environment variables
load_dotenv(find_dotenv())

llm = OpenAI(temperature=0)

        # steps_eda()


# Set the title of the Streamlit app
st.title('AI Data Analysis Assistant by WinOps 🤖')
# Welcome message to the user
st.write("Hello 👋🏻 I am your WinOps AI Assistant, and I am here to help you with your data analysis problems.")

# Set up the sidebar
with st.sidebar:
    st.write(
        '''
        ## Steps to Follow
        1. **Upload Your Data**: Start by uploading your dataset to our platform. We support both CSV and XLSX file formats.
        2. **Choose Your Task**: Select the task you want to perform on your data. This could be anything from data cleaning to visualization.
        3. **Get Your Solution**: Our AI Assistant will then work its magic to provide you with a solution tailored to your task. This might include data insights, visualizations, or even code snippets to help you achieve your goals.
        '''
    )
    st.caption('''
    Happy Learning!
    ''')

    st.divider()

    st.markdown("<p style='text-align: center;'>Powered by WinOps</p>", unsafe_allow_html=True)
# if st.button("Let's get started"):
#     pass
    # Initialise a session state variable
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True

# Button with callback function
st.button("Let's get started", on_click=clicked, args=[1])
    # Action to perform when the button is clicked 
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here!", type=["csv", "xlsx"])  # Accept both CSV and XLSX files
    
    if user_csv is not None:
        user_csv.seek(0)
        if user_csv.name.endswith('.csv'):
            df = pd.read_csv(user_csv, low_memory=False)
        elif user_csv.name.endswith('.xlsx'):
            df = pd.read_excel(user_csv, engine='openpyxl')  # Use openpyxl engine for xlsx files
        # Initialize the LangChain model with a temperature of 0
        llm = OpenAI(temperature=0)

        @st.cache_data
        def steps_eda():
            steps = llm("What are the steps of EDA?")
            return steps
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

        @st.cache_data
        def function_agent():
            from concurrent.futures import ThreadPoolExecutor

            def process_task(task):
                try:
                    return pandas_agent.run(task)
                except RateLimitError:
                    st.warning("Rate limit exceeded. Please consider changing your API key.")
                    return None

            tasks = [
                "What are the meaning of the columns?",
                "How many missing values does this dataframe have? Start the answer with 'There are'",
                "Are there any duplicate values and if so where?",
                "Calculate correlations between numerical variables to identify potential relationships.",
                "Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.",
                "What new features would be interesting to create?."
            ]

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(process_task, tasks))

            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            st.write(results[0])
            st.write(results[1])
            st.write(results[2])
            st.write("**Data Summarisation**")
            st.write(df.describe())
            st.write(results[3])
            st.write(results[4])
            st.write(results[5])
            return
        @st.cache_data
        def function_question_variable():
            # Display column headings for user selection and provide extra information
            column_headings = df.columns.tolist()
            st.write("Please refer to the column headings below to select a variable:")
            st.write(column_headings)
            user_question_variable = st.selectbox("Select a variable:", column_headings)
            
            st.line_chart(df, y=user_question_variable)
            tasks = [
                "What are the summary statistics of the variable " + user_question_variable + "?",
                "What is the distribution of the variable " + user_question_variable + "?",
                "What are the outliers of the variable " + user_question_variable + "?",
                "What are the trends of the variable " + user_question_variable + "?"
            ]

            from concurrent.futures import ThreadPoolExecutor

            def process_task(task):
                try:
                    result = pandas_agent.run(task)
                    return result
                except RateLimitError:
                    st.write("Rate limit exceeded. Please consider changing your API key.")
                    return None

            with ThreadPoolExecutor(max_workers=1) as executor:
                results = list(executor.map(process_task, tasks))

            for result in results:
                if result is not None:
                    st.write(result)
            return

        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_variable_further)
            st.write(dataframe_info)
            return

        #Main
        st.header('Exploratory Data Analysis')
        st.subheader("General Information about the dataset")
        with st.sidebar:
            with st.expander("Steps are the steps of EDA"):
                st.write(steps_eda())
        function_agent()
        # st.subheader("Variable of the study ")
        # Get user input
        user_question_variable = st.text_input("What variables are you interested in?")
        if user_question_variable is not None and user_question_variable != "":
            try:
                # Attempt to use the user input
                pass
            except Exception as e:
                st.error(f"Error: {e}")

            # function_question_variable()
            st.subheader("Further study")
        
        if user_question_variable:
            user_question_variable_further = st.text_input("What further question do you have?")
            if user_question_variable_further is not None and user_question_variable_further and user_question_variable_further not in("", "no", "No", "NO"):
                function_question_dataframe()
                #st.write(pandas_agent.run(user_question_variable_further))
            if user_question_variable_further is ("no", "No"):
                st.write("Thank you for using WinOps AI Assistant!")
        # st.subheader("Solution")
                if user_question_variable_further:
                    st.divider()
                    st.header("Data Science Problem")
                    st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's time to answer business intelligence questions.")
        # function_agent()
                    prompt = st.text_input("What is the problem you want to solve?")
                 
                    
                    data_problem_template = PromptTemplate(
                        input_variables=['business_problem'],
                        template='Convert the following business problem into a data science problem: {business_problem}.'
                    )
                    data_problem_chain = LLMChain(llm=llm, prompt_template=data_problem_template, verbose=True)
                    
                    if prompt: 
                        response = data_problem_chain.run(business_problem=prompt)
                        st.write(response)



with st.sidebar:    
    with st.expander("About"):
        st.write(llm("What are the steps of EDA?"))
