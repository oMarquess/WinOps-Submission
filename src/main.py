# Import necessary modules and set up the OpenAI API key
import os
from apikey import openai_api_key

import streamlit as st
import pandas as pd

from langchain.llms import OpenAI
#from langchain.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
# Set the OpenAI API key from the environment variable
os.environ['OPENAI_API_KEY'] = openai_api_key
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
        ## Steps
        - Upload your data
        - Choose the task
        - Get the solution
        '''
    )
    st.caption('''
    Lorem Ipsum
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
    user_csv = st.file_uploader("Upload your file here!", type="csv")
    
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)
        # Initialize the LangChain model with a temperature of 0
        llm = OpenAI(temperature=0)

        @st.cache_data
        def steps_eda():
            steps = llm("What are the steps of EDA?")
            return steps
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose=True)



        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return

# function_agent()

# if st.button("Let's get started", key='button1'):
#     st.write("Hello 👋🏻 I am your WinOps AI Assistant, and I am here to help you with your data analysis problems.")




# if st.button("Let's get started", key='button2'):
#     st.header('Exploratory Data Analysis')
#     st.subheader("Solution")


        #st.write(df.head())

# Set up the About section in the sidebar
with st.sidebar:    
    with st.expander("About"):
        st.write(llm("What are the steps of EDA?"))

# function_agent()

# Get user input
user_question = st.text_input("What would you like to know about the data?")

# if user_question:
#     st.write(pandas_agent.run(user_question))


# st.header('Exploratory Data Analysis')
# st.subheader("Solution")



# llm = OpenAI(temperature=0)

# steps_eda = llm("What steps are involved in EDA?")
