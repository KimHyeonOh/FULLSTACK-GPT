'''
풀스택 GPT 챌린지 Assignment 6
gpt-challenge4.ipynb 참조
이전 과제에서 구현한 RAG 파이프라인을 Streamlit으로 마이그레이션합니다.
파일 업로드 및 채팅 기록을 구현합니다.
사용자가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
st.sidebar를 사용하여 스트림릿 앱의 코드와 함께 깃허브 리포지토리에 링크를 넣습니다.
'''

from typing import Dict, List
import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

if "messages" not in st.session_state:
    st.session_state["messages"] = []


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def initialize_llm(api_key: str):
    if not api_key:
        st.error("OpenAI API Key가 필요합니다.")
        return None
    return ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        api_key=api_key
    )

# llm = ChatOpenAI(
#     temperature=0.1,
#     streaming=True,
#     callbacks=[
#         ChatCallbackHandler(),
#     ]
# )


@st.cache_data(show_spinner="파일 임베딩중...")
def embed_file(file, api_key):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    dir_path = os.path.dirname(file_path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(file_path, "wb") as f:
        f.write(file_content)

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir,
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
    Answer the question using ONLY the following context. If you don't know the answer just say yhou don't know DON'T make anything up.

    Context: {context}
    """
     ),
    ("human", "{question}"),
])

st.title("풀스택 GPT 챌린지 Assignment 6")

st.markdown("""
사이드바의 파일업로드버튼을 이용하여 AI에게 질문하세요.
""")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key를 입력하세요.", type="password")
    file = st.file_uploader(
        "Upload a .txt .pdf or .hwp file",
        type=["pdf", "txt", "hwp"],
    )

if api_key:
    os.environ['OPENAI_API_KEY'] = api_key

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
        api_key=api_key
    )

if file:
    if not api_key:
        st.error('OpenAI API Key가 필요합니다.')
    else:
        retriever = embed_file(file, api_key)
        send_message("임베딩이 완료되었습니다. 질문해주세요.", "ai", save=False)
        paint_history()
        message = st.chat_input("등록한 파일에 대해 궁금한 것을 질문하세요.")
        if message:
            send_message(message, "human")
            chain = {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            } | prompt | llm
            # docs = retriever.invoke(message)
            # docs = "\n\n".join(document.page_content for document in docs)
            # prompt = template.format_messages(context=docs, question=message)
            # llm.predict_message(prompt)
            # response = chain.invoke(message)
            # send_message(response.content, "ai")
            with st.chat_message("ai"):
                response = chain.invoke(message)
else:
    st.session_state["messages"] = []
