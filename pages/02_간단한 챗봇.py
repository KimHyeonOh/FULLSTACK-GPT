import streamlit as st
import time

st.title("PrivateGPT")

########################################################################################################################

# st.title("Document GPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


for message in st.session_state["messages"]:
    send_message(message["message"], message["role"], save=False)

message = st.chat_input("AI에게 보낼 문장을 입력하세요.")

if message:
    send_message(message, "human")
    time.sleep(1)
    send_message(f"You said: {message}", "ai")

    with st.sidebar:
        st.write(st.session_state)

########################################################################################################################

# with st.chat_message("human"):
#     st.write("Hi~")

# with st.chat_message("ai"):
#     st.write("how are you?")

# with st.status("Embedding file...", expanded=True) as status:
#     st.write("Getting the file")
#     time.sleep(3)
#     st.write("Embedding the file")
#     time.sleep(3)
#     st.write("Caching the file")
#     status.update(label="Error", state="error")

########################################################################################################################
