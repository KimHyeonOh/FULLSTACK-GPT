import streamlit as st
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


def get_weather(lon, lat):
    print("call an api....")


function = {
    "name": "get_weather",
    "description": "function that takes longitude and latitude to find the weather of a place",
    "parameters": {
        "type": "object",
        "properties": {
            "lon": {
                "type": "string",
                "description": "The longitude coordinate"
            },
            "lat": {
                "type": "string",
                "description": "The latitude coordinate"
            }
        }
    },
    "required": ["lon", "lat"]
}

llm = ChatOpenAI(temperature=0.1).bind(
    function_call="auto", functions=[function])
prompt = PromptTemplate.from_template("Who is the weather in {city}")

chain = prompt | llm

response = chain.invoke({"city": "seoul"})
response = response.additional_kwargs["function_call"]["arguments"]

# r = json.loads(response)
# weather = get_weather(r['lon'], r['lat'])
# st.markdown(weather)

st.markdown(response)
