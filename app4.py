import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.schema import SystemMessage
from pydantic import BaseModel, Field
from typing import Type
from langchain.tools import BaseTool
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import WebBaseLoader
from langchain.tools import DuckDuckGoSearchResults
from langchain.tools import WikipediaQueryRun

# Define tools
class DuckDuckGoSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for")

class DuckDuckGoSearchTool(BaseTool):
    name = "DuckDuckGoSearchTool"
    description = """
    Use this tool to perform web searches using the DuckDuckGo search engine.
    It takes a query as an argument.
    Example query: "Latest technology news"
    """
    args_schema: Type[DuckDuckGoSearchToolArgsSchema] = DuckDuckGoSearchToolArgsSchema

    def _run(self, query) -> str:
        search = DuckDuckGoSearchResults()
        return search.run(query)

class WikipediaSearchToolArgsSchema(BaseModel):
    query: str = Field(description="The query you will search for on Wikipedia")

class WikipediaSearchTool(BaseTool):
    name = "WikipediaSearchTool"
    description = """
    Use this tool to perform searches on Wikipedia.
    It takes a query as an argument.
    Example query: "Artificial Intelligence"
    """
    args_schema: Type[WikipediaSearchToolArgsSchema] = WikipediaSearchToolArgsSchema

    def _run(self, query) -> str:
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wiki.run(query)

class WebScrapingToolArgsSchema(BaseModel):
    url: str = Field(description="The URL of the website you want to scrape")

class WebScrapingTool(BaseTool):
    name = "WebScrapingTool"
    description = """
    If you found the website link in DuckDuckGo,
    Use this to get the content of the link for my research.
    """
    args_schema: Type[WebScrapingToolArgsSchema] = WebScrapingToolArgsSchema

    def _run(self, url):
        loader = WebBaseLoader([url])
        docs = loader.load()
        text = "\n\n".join([doc.page_content for doc in docs])
        return text

class SaveToTXTToolArgsSchema(BaseModel):
    text: str = Field(description="The text you will save to a file.")

class SaveToTXTTool(BaseTool):
    name = "SaveToTXTTOOL"
    description = """
    Use this tool to save the content as a .txt file.
    """
    args_schema: Type[SaveToTXTToolArgsSchema] = SaveToTXTToolArgsSchema

    def _run(self, text) -> str:
        with open("research_results.txt", "w") as file:
            file.write(text)
        return "Research results saved to research_results.txt"

# Streamlit app
def main():
    st.set_page_config(page_title="OpenAI Assistant", layout="wide")

    # Sidebar for API Key and GitHub link
    with st.sidebar:
        st.title("Settings")
        api_key = st.text_input("Enter your OpenAI API Key", type="password")

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
        return

    # Set up agent
    llm = ChatOpenAI(temperature=0.1, model="gpt-4", openai_api_key=api_key)

    system_message = SystemMessage(
        content="""
            You are a research expert.

            Your task is to use Wikipedia or DuckDuckGo to gather comprehensive and accurate information about the query provided. 

            When you find a relevant website through DuckDuckGo, you must scrape the content from that website. Use this scraped content to thoroughly research and formulate a detailed answer to the question. 

            Combine information from Wikipedia, DuckDuckGo searches, and any relevant websites you find. Ensure that the final answer is well-organized and detailed, and include citations with links (URLs) for all sources used.

            Your research should be saved to a .txt file, and the content should match the detailed findings provided. Make sure to include all sources and relevant information.

            The information from Wikipedia must be included.

            Ensure that the final .txt file contains detailed information, all relevant sources, and citations.
        """
    )

    agent = initialize_agent(
        llm=llm,
        verbose=True,
        agent=AgentType.OPENAI_FUNCTIONS,
        tools=[
            DuckDuckGoSearchTool(),
            WikipediaSearchTool(),
            WebScrapingTool(),
            SaveToTXTTool(),
        ],
        agent_kwargs={"system_message": system_message},
    )

    # Main UI
    st.title("OpenAI Assistant with Research Tools")
    query = st.text_input("Enter your query:")

    if query:
        with st.spinner("Processing your request..."):
            try:
                results = agent.run(query)
                st.success("Research completed successfully!")
                st.text_area("Results:", value=results, height=400)
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
