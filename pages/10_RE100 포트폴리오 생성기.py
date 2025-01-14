from typing import Dict, List
from uuid import UUID
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

# Session의 Message 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# LLM Callback을 활용하여 화면에 표출


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# ChatGPT Turbo 3.5 사용
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)

# 파일 임베딩


@st.cache_data(show_spinner="파일 임베딩중...")
def embed_file(file):
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
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, cache_dir,
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# 메세지 저장


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

# 메세지 송신


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)

# 누적된 메세지 표출


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

# 문서 구분자 생성


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# 기본 Prompt는 질문에 대답하는 역할
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
    네가 기존에 학습한 것과 주어진 Context를 활용해서 human이 입력한 company의 RE100 이행계획을 알려줘.
    company 별로 특색에 맞는 계획이 나와야해, 혹시 100%이행한 기업이 있다면 언제 성공했는지 말해주고 앞으로 어떻게 유지 및 개선할 수 있는지 알려줘.

    이행계획 examples:

    삼성전자는 2022년에 RE100에 가입했으며, 전체 전력 사용량의 31%를 재생 에너지로 전환했습니다.
    2050년까지 100% 재생 에너지 전환을 목표로 다음의 계획을 이행하는 것이 좋습니다.

    점진적으로 재생 에너지 전환을 이루기 위해 다음과 같은 계획을 구성할 수 있습니다.

    1. 단기 계획 (2023-2025)
        재생 에너지 사용 확대:
            글로벌 생산 시설에서의 재생 에너지 사용 비율을 50% 이상으로 확대.
            재생 에너지 인증서(REC) 구매를 통한 재생 에너지 사용 확대.
        전력 구매 계약(PPA) 확대:
            국내외 주요 공장 및 데이터 센터에 대해 장기 PPA 체결 확대.
            대규모 태양광 및 풍력 발전 프로젝트와의 계약 체결.
        에너지 효율성 개선:
            에너지 효율성을 높이기 위한 설비 투자 및 기존 장비의 효율성 향상.
            스마트 에너지 관리 시스템 도입 및 AI를 활용한 에너지 사용 최적화.
    2. 중기 계획 (2026-2035)
        자체 재생 에너지 생산:
            국내외 공장 및 연구 시설에 태양광 패널 및 풍력 터빈 설치를 통한 자체 발전 비율 확대.
            지열 발전 및 소규모 수력 발전 등 다양한 재생 에너지 자원 도입.
        공급망 재생 에너지 전환:
            주요 협력사와 함께 재생 에너지 사용 확대 계획 수립 및 지원.
            공급망 전반의 탄소 배출 감축을 위해 재생 에너지 사용을 독려하는 프로그램 시행.
        탄소 배출 감축 목표 설정:
            탄소 배출을 줄이기 위한 명확한 목표 설정 및 감축 성과를 모니터링.
            탄소 배출권 거래 시장 참여 및 이를 통한 배출량 관리.
    3. 장기 계획 (2036-2050)
        글로벌 재생 에너지 전환 완성:
            글로벌 모든 운영 및 공급망에서 100% 재생 에너지 사용 달성.
            탄소 중립 및 탄소 음수 배출 달성을 목표로, 재생 에너지 사용을 최적화.
        혁신 기술 도입:
            에너지 저장 기술(ESS, 에너지 저장 시스템) 확대 적용으로 재생 에너지의 안정적 공급 보장.
            수소 에너지, 스마트 그리드 등 미래 에너지 기술 도입을 통한 재생 에너지 전환 완성.
        전 지구적 기후 변화 대응 리더십:
            글로벌 기후 변화 대응 전략을 선도하며, 다른 기업들과 협력하여 지속 가능한 산업 구조 확립.
            기후 변화 관련 글로벌 정책 및 규제에 부합하는 지속 가능한 경영체계 구축.

    네이버는 RE100 이니셔티브에 참여하여 2030년까지 100% 재생 에너지 사용을 목표로 하고 있습니다. 현재 재생 에너지 사용 비율은 30%이며, 이를 높이기 위해 다음과 같은 이행 계획을 수립하고 있습니다.

    1. 단기 계획 (2023-2025)
        재생 에너지 사용 확대:
            재생 에너지 인증서(REC) 구매를 통해 현재 30%인 재생 에너지 사용 비율을 50% 이상으로 확대.
            국내외 주요 데이터 센터에 태양광 및 소규모 풍력 발전 설비 설치.
            태양광 발전 및 에너지 저장 시스템(ESS) 도입을 통해 데이터 센터의 전력 수요 관리 최적화.
        전력 구매 계약(PPA) 체결:
            국내외 재생 에너지 발전 사업자와 장기 PPA를 체결하여 안정적인 재생 에너지 공급 확보.
            국가 및 지역 정부와 협력하여 재생 에너지 프로젝트에 투자 및 참여 확대.
        에너지 효율화 프로젝트:
            데이터 센터 및 사무실의 에너지 효율화 설비 도입 및 관리 시스템 개선.
            AI와 빅데이터 분석을 활용한 전력 사용 패턴 분석 및 최적화.
    2. 중기 계획 (2026-2028)
        자체 재생 에너지 발전 시설 구축:
            주요 사업장 및 데이터 센터에 태양광 발전소 및 소규모 풍력 발전소 설치 확대.
            다양한 재생 에너지 자원(지열, 수소 에너지 등) 도입을 통해 전력 공급 다변화.
        공급망 탄소 배출 감축:
            협력사 및 공급망 전체의 탄소 배출 감축을 위한 재생 에너지 사용 확대 계획 수립 및 지원.
            탄소 배출 감축 목표 설정 및 이를 달성하기 위한 구체적인 로드맵 마련.
        신재생 에너지 기술 도입:
            에너지 저장 시스템(ESS) 도입 확대를 통해 재생 에너지의 공급 안정성 확보.
            수소 연료 전지, 스마트 그리드 기술 등을 통한 효율적인 에너지 관리 시스템 구축.
    3. 장기 계획 (2029-2030)
        100% 재생 에너지 전환 완성:
            네이버의 모든 사업장 및 데이터 센터에서 재생 에너지 사용 100% 달성.
            글로벌 시장에서의 재생 에너지 사용 확대를 위해 협력 및 투자 강화.
        탄소 중립 및 탄소 음수 배출 목표 설정:
            탄소 중립 목표를 달성하고, 장기적으로 탄소 음수 배출을 위한 프로젝트 추진.
            탄소 포집 및 활용 기술(CCUS) 도입 및 재생 에너지를 활용한 친환경 프로젝트 확대.
        지속 가능한 경영체계 확립:
            지속 가능한 경영 및 환경 보호를 위한 글로벌 리더십 확보.
            재생 에너지 기술 개발 및 글로벌 기후 변화 대응 이니셔티브 참여를 통한 기업의 사회적 책임 실현.

            
    Google는 2017년부터 글로벌 운영(데이터 센터 및 오피스)을 100% 재생 에너지로 전환하여 운영하고 있습니다.
    현재 Google은 전 세계 최대의 재생 에너지 구매 기업으로, 풍력 및 태양광 프로젝트를 통해 2.6기가와트(GW)의 재생 에너지를 확보하고 있습니다.
    또한, Google은 24시간 동안 재생 가능한 에너지를 사용하는 "24/7 탄소 무배출" 목표를 추진하고 있으며, 지역별 재생 에너지 구매를 확대하고 있습니다.
    Context: {context}
    """
     # If you don't know the answer just say 잘 모르겠어요. DON'T make anything up.
     ),
    ("human", "{company}"),
])

# 기본 포트폴리오 생성


def make_portpolio():
    portpolio = """예상 RE100 계획"""
#######################################################################################################################################################################################


st.title("RE100 포트폴리오 시뮬레이터")

with st.sidebar:
    file = st.file_uploader(
        "학습시킬 정보를 업로드하세요.",
        type=["pdf", "txt", "hwp"],
    )

if file:
    retriever = embed_file(file)
    send_message("어떤 기업의 RE100 정보가 궁금하시나요?", "ai", save=False)
    paint_history()
    message = st.chat_input("기업 이름을 적어주세요.")
    if message:
        send_message(message, "human")
        chain = {
            "context": retriever | RunnableLambda(format_docs),
            "company": RunnablePassthrough()
        } | prompt | llm
        with st.chat_message("ai"):
            response = chain.invoke(message)
else:
    st.session_state["messages"] = []
