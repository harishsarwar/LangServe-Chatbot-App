from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from fastapi import FastAPI
from langserve import add_routes
import os
import uvicorn


api_key = os.getenv("HUGGINGFACE_API_KEY")

os.environ["HUGGINGFACE_API_KEY"]=api_key 

repo_id = "microsoft/Phi-3.5-mini-instruct"

llm=HuggingFaceEndpoint(repo_id=repo_id,
                        max_new_tokens=1000, 
                        temperature=0.7,
                        do_sample=True,
                        repetition_penalty=1.1,
                        return_full_text=True,
                        huggingfacehub_api_token=api_key)


app = FastAPI(
    title="demo app for testing",
    description="demo for langserve",
    version= "1.0",

)


prompt1 = PromptTemplate.from_template("make story without adding points {topic}")

add_routes (
    app,
    prompt1 | llm,
    path= "/story",
)

prompt2 = PromptTemplate.from_template("what is topic is given explain that topic in pointswise {topic} ")
add_routes(
    app,
    prompt2 | llm,
    path= "/topic",
)

if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8000)