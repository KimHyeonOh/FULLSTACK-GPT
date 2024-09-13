import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="FullstackGPT Home",
    page_icon="♪♪♪"
)

st.title("FullstackGPT Home")

# today = datetime.today().strftime("%H:%M:%S")

# st.title("최종 수정 시각\t" + today)

# model = st.selectbox(
#     "Choose your model",
#     ("GPT-3","GPT-4",)
# )

# if model == "GPT-3":
#     st.write("저렴한 모델")
# else:
#     st.write("비싼 모델")

# name = st.text_input("당신의 이름을 입력해주세요.")
# st.write(name)

# value = st.slider("temperature", min_value=0.1, max_value=1.0,)
# st.write(value)

# with st.sidebar:
#     st.sidebar.title("sidebar title")
#     st.sidebar.text_input("xxx")

# st.title("title")

# tab_one, tab_two, tab_three = st.tabs(["A", "B", "C"])

# with tab_one:
#     st.write("a")

# with tab_two:
#     st.write("b")

# with tab_three:
#     st.write("c")