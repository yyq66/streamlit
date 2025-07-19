
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3

import streamlit as st
from zhipuai_embedding import ZhipuAIEmbeddings
from zhipuai_llm import ZhipuaiLLM
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableBranch
from langchain_core.output_parsers import StrOutputParser
import os

def get_retriever():
    """
    获取检索器
    """
    embeddings = ZhipuAIEmbeddings()
    # 向量库持久化路径
    persist_directory = './data_base/vector_db/chroma'
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    # 返回检索器
    return vectordb.as_retriever()

def combine_docs(docs):
    """
    处理检索器返回的文本：将文档内容组合成一个字符串
    """
    return "\n\n".join([doc.page_content for doc in docs['context']])

def get_qa_history_chain():
    """ 
    获取一个问答历史链
    """
    retriever = get_retriever()
    llm = ZhipuaiLLM(model_name="glm-4-plus", temperature=0.1, api_key=os.getenv("ZHIPUAI_API_KEY"))

    # 压缩问题的系统提示模板
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )

    # 构造 压缩问题的 prompt template
    continue_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    # 构造检索文档的链
    qa_chain = (
        RunnablePassthrough.assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    # 构造检索文档的链
    # RunnableBranch 会根据条件选择要运行的分支
    retriever_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x['input']) | retriever),
        continue_question_prompt | llm | StrOutputParser() | retriever
    )

    qa_history_chain = RunnablePassthrough.assign(
        context = retriever_docs
    ).assign(answer=qa_chain)

    return qa_history_chain

def gen_response(chain, input, chat_history):
    """
    接受检索问答链、用户输入及聊天历史，并以流式返回该链输出
    """
    response = chain.stream({
        'input': input,
        "chat_history": chat_history
    })

    for res in response:
        if "answer" in res.keys():
            yield res["answer"]


def main():
    st.title("👋 Hello, Streamlit!")







def main():
    """
    Streamlit 应用的主函数:
        该函数制定显示效果与逻辑
    """
    st.markdown('### 🦜🔗 动手学大模型应用开发')
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages: # 遍历对话历史
            with messages.chat_message(message[0]): # messages指在容器下显示，chat_message显示用户及ai头像
                st.write(message[1]) # 打印内容
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))

main()




# res = get_qa_history_chain().invoke({
#     "input": "南瓜书跟它有什么关系？",
#     "chat_history": [
#         ("human", "西瓜书是什么？"),
#         ("ai", "西瓜书是指周志华老师的《机器学习》一书，是机器学习领域的经典入门教材之一。"),
#     ]
# })
# print("res:", res)