import os
import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv
from openai import RateLimitError
from concurrent.futures import ThreadPoolExecutor
from langchain_experimental.agents import create_pandas_dataframe_agent

# Load environment variables
load_dotenv(find_dotenv())

# Set the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

# Initialize the LangChain model
llm = OpenAI(temperature=0)

# Set the title of the Streamlit app
st.title('AI Data Analysis Assistant by WinOps 🤖')
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
    st.caption('Happy Analysis!')
    st.divider()
    st.markdown("<p style='text-align: center;'>Powered by WinOps</p>", unsafe_allow_html=True)

# Initialize session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])

# Function to display EDA steps
@st.cache_data
def steps_eda():
    try:
        return llm("What are the steps of EDA?")
    except RateLimitError as e:
        st.warning(f"Rate limit reached. Please try again later. Error details: {e}")
        return None

# Function to process data analysis tasks
def process_task(task, pandas_agent):
    try:
        return pandas_agent.run(task)
    except RateLimitError:
        st.warning("Rate limit exceeded. Please consider changing your API key.")
        return None

# Function to perform exploratory data analysis
def function_agent(pandas_agent, df):
    tasks = [
        "What are the meaning of the columns?",
        "How many missing values does this dataframe have? Start the answer with 'There are'",
        "Are there any duplicate values and if so where?",
        "Calculate correlations between numerical variables to identify potential relationships.",
        "Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.",
        "What new features would be interesting to create?"
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda task: process_task(task, pandas_agent), tasks))

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

# Function to analyze a specific variable
def function_question_variable(df, pandas_agent, user_question_variable):
    st.line_chart(df, y=user_question_variable)
    tasks = [
        f"What are the summary statistics of the variable {user_question_variable}?",
        f"What is the distribution of the variable {user_question_variable}?",
        f"What are the outliers of the variable {user_question_variable}?",
        f"What are the trends of the variable {user_question_variable}?"
    ]

    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(lambda task: process_task(task, pandas_agent), tasks))

    for result in results:
        if result:
            st.write(result)

# Function to handle further questions on the dataframe
def function_question_dataframe(pandas_agent, user_question_variable_further):
    dataframe_info = pandas_agent.run(user_question_variable_further)
    st.write(dataframe_info)

if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here!", type=["csv", "xlsx"])  # Accept both CSV and XLSX files
    
    if user_csv is not None:
        user_csv.seek(0)
        if user_csv.name.endswith('.csv'):
            df = pd.read_csv(user_csv, low_memory=False)
        elif user_csv.name.endswith('.xlsx'):
            df = pd.read_excel(user_csv, engine='openpyxl')  # Use openpyxl engine for xlsx files

        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)

        # st.header('Exploratory Data Analysis')
        st.subheader("General Information about the dataset")
        
        with st.sidebar:
            with st.expander("Steps of EDA"):
                steps = steps_eda()
                if steps:
                    st.write(steps)

        col1, col2, col3 = st.columns(3)
        with col1:
            run_data_analysis = st.button("Run Exploratory Data Analysis")
        with col2:
            analyze_variable = st.button("Analyze Variable")
        with col3:
            further_study = st.button("Business Intel Questions")
        
        if run_data_analysis:
            function_agent(pandas_agent, df)
        
        if analyze_variable:
            column_headings = df.columns.tolist()
            user_question_variable = st.selectbox("Select a variable:", column_headings)
            if user_question_variable:
                function_question_variable(df, pandas_agent, user_question_variable)
        
        if further_study:
            user_question_variable = st.text_input("What variables are you interested in?")
            if user_question_variable:
                user_question_variable_further = st.text_input("What further question do you have?")
                if user_question_variable_further and user_question_variable_further.lower() not in ("no", "no", "no"):
                    function_question_dataframe(pandas_agent, user_question_variable_further)
                if user_question_variable_further.lower() in ("no", "no"):
                    st.write("Thank you for using WinOps AI Assistant!")
                    
                if user_question_variable_further:
                    st.divider()
                    st.header("Data Science Problem")
                    st.write("Now that we have a solid grasp of the data at hand and a clear understanding of the variable we intend to investigate, it's time to answer business intelligence questions.")
                    
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
        st.write(steps_eda())
