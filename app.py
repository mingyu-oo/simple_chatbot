############################################################################
#  streamlit/05_streamlit_chat_exam_session_state_llm_streaming_memory.py
############################################################################
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


# API KEY 불러오기.
load_dotenv()

# LLM model 생성
@st.cache_resource  # cache에 올리겠다는 뜻. cache에 올라가 있으니 rerun해도 이 함수를 실행하지 않음. 다시 껐다 켜야함
def get_llm_model():
    load_dotenv()
    model = ChatOpenAI(model_name = "gpt-4o-mini")
    prompt_template = ChatPromptTemplate(
    messages = [
        MessagesPlaceholder(variable_name = "history", optional = True),    # 대화 이력
        ("user", "{query}")    # 사용자 입력
        ]
    )
    return prompt_template | model | StrOutputParser()

# PromptTemplate로 정의해도 잇더즌매터.
# """
# # Instruction
# {query}
# 답변에 대해서 응답해주세요.

# # Context
# {history}

# # Input
# {query}

# """


model = get_llm_model()   # cache에 올려져 있는걸 사용, chain이 넘어옴.

st.title("Chatbot + Session State + LLM 연동 튜토리얼")


# Session State 생성
## session_state : dictionary 구현체, 시작 ~ 종료할 때 까지 사용자 별로 유지되어야 하는 값들을 저장하는 곳

# 0. 대화 내욕을 session_state의 "messages" : list 로 저장
# 1. session state에 messages key 조회(없으면 생성)
if "messages" not in st.session_state:   # return T/F
    st.session_state["messages"] = []    # 대화 내용들을 저장할 list를 "messages" 키로 저장


# 기존 대화 이력 출력
message_list = st.session_state["messages"] # 변수로 저장.
# history_message_list = [(msg_dict["role"], msg_dict["content"]) for msg_dict in message_list]
# message_list = [{"role" : "user", "content" : "입력내용"}]
# -> history_message_list = [("user", "입력내용")]  template에 tuple형태로 들어가야 하기 때문에 변경.
# MessagesPlaceholder(ChatPromptTemplate의 messages 형식)에 입력 형식 ("user", "{query}")

for message in message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# user의 prompt를 입력 받는 위젯
prompt = st.chat_input("User Prompt")    # user가 입력한 문자열을 반환.


## 대화 작업
if prompt is not None:
    # session_state의 messages에 대화 내역을 저장
    st.session_state["messages"].append({"role" : "user", "content" : prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("ai"):
        message_placeholder = st.empty()    # update가 가능한 container
        full_message = ""   # LLM이 응답하는 토큰들을 저장할 문자열 변수.
        for token in model.stream({"query" : prompt, "history" : message_list}):
            full_message += token
            message_placeholder.write(full_message) # 기존 내용을 full_message로 갱신.
        st.session_state["messages"].append({"role" : "ai", "content" : full_message})






# # 대화 내역을 chat_message container에 출력
# for message_dict in st.session_state["messages"]:
#     with st.chat_message(message_dict["role"]):
#         st.write(message_dict["content"])


# [
#     {
#     "role" : "user", "content" : prompt
#     },
#     {
#     "role" : "AI", "content" : ai_message
#     }
# ]