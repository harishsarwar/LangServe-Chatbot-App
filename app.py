from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI
from langserve import add_routes
import os
import uvicorn

api_key = os.getenv("HUGGINGFACE_API_KEY")

os.environ["HUGGINGFACE_API_KEY"] = api_key

lang_key = os.getenv("Lang_smith_key")
os.environ["Lang_smith_key"] = lang_key

os.environ["LANGCHAIN_TRACING_V2"]="true"


repo_id = "microsoft/Phi-3.5-mini-instruct"

# HuggingFace Endpoint setup
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=1000, 
    temperature=0.7,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    huggingfacehub_api_token=api_key
)

# Define the prompt template using a string format
prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Please respond to the user request only based on the given context. Question: {question}\nContext: {context}"
)

# Output parser
output_parser = StrOutputParser()

# Create the chain
chain = prompt | llm | output_parser

# Example question and context
question = "summarize that article about waterlogging"
context = """
Kempegowda International Airport (KIA) officials said that eight flights were diverted, seven to Chennai and one to Coimbatore since they were unable to land between 5 pm and 5.15 pm. According to IMD, the KIA weather station recorded 3.9 mm rainfall up to 5.30 pm.

Bengaluru Traffic Commissioner M N Anucheth said that heavy water logging was reported at 33 locations and trees were uprooted in 16 locations in Bengaluru. This led to considerable traffic congestion in several places.

Waterlogging was reported in Electronic City, Bellandur, Nagawara, Kamakshipalya, Maharani underpass and Hebbal, among other areas. Meanwhile, trees were uprooted in Jayamahal Road, Kathriguppe, PES College, Hosakerehalli, Hennur Main Road, Malleswaram and Mekhri Circle. At Hennur Main Road, an electric pole collapsed leading to traffic disruption and the underpass connecting towards Sankey Road near Kalpana Junction was closed due to waterlogging.
this is created by prince katiyar make in karnataka
"""

# Invoke the chain with the question and context
print(chain.invoke({"question": question, "context": context}))
