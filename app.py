import os
import streamlit as st
from langchain.chains import create_sql_query_chain
from langchain_google_genai import GoogleGenerativeAI
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv

load_dotenv()

db_user = "root"
db_password = "Nafra0595."
db_host = "localhost"
db_name = "classicmodels"

engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

db = SQLDatabase(engine, sample_rows_in_table_info=3)

# Few Shot Fine Tuning
few_shot_examples = """
Example 1:
User: What is the price of the product with ID 1?
SQL Query: SELECT priceEach FROM products WHERE productCode = 'S10_1111';

Example 2:
User: Show me the product names and prices for all products in category 2.
SQL Query: SELECT productName, priceEach FROM products WHERE productCode IN (SELECT productCode FROM productlines WHERE productLine = 'Classic Cars');

Example 3:
User: List the products whose price is greater than 50.
SQL Query: SELECT productName, priceEach FROM products WHERE priceEach > 50;

Example 4:
User: What is the total stock quantity for all products in category 'Motorcycles'?
SQL Query: SELECT SUM(quantityInStock) FROM products WHERE productCode IN (SELECT productCode FROM productlines WHERE productLine = 'Motorcycles');
"""

# Initialize LLM with a custom prompt template
prompt_template = PromptTemplate(
    input_variables=["input", "top_k", "table_info", "few_shot_examples"],
    template=(  
        "Given the following table structure and sample data:\n"
        "{table_info}\n\n"
        "Based on the user's input:\n"
        "{input}\n\n"
        "Here are some example questions and their corresponding SQL queries:\n"
        "{few_shot_examples}\n\n"
        "The SQL query should be syntactically correct and optimized for MySQL. "
        "Ensure that the query is efficient and considers returning up to {top_k} rows if applicable.\n"
        "Your response should be in the form of a valid SQL query. "
        "Please ensure that the query matches the user's intent precisely."
    )
)


llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

#SQL query chain with custom prompt
chain = create_sql_query_chain(llm, db, prompt=prompt_template)

def execute_query(question):
    try:
        dynamic_few_shot = few_shot_examples 
        
        response = chain.invoke({
            "question": question,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "few_shot_examples": dynamic_few_shot
        })

        cleaned_query = response.strip("```sql").strip()

        result = db.run(cleaned_query)
                
        return cleaned_query, result
    except ProgrammingError as e:
        st.error(f"An error occurred: {e}")
        return None, None

st.title("SQL Query Chatbot for Vehicles using Gemini API 🏎️🚕")

question = st.text_input("Enter your question:")

if st.button("Execute"):
    if question:
        cleaned_query, query_result = execute_query(question)
        
        if cleaned_query and query_result is not None:
            st.write("Generated SQL Query:")
            st.code(cleaned_query, language="sql")
            st.write("Query Result:")
            st.write(query_result)
        else:
            st.write("No result returned due to an error.")
    else:
        st.write("Please enter a question.")
