#存算分离,app.py 负责在线问答（快速响应）
#要先在cmd中运行ollama run qwen2.5:7b
#在终端输入：streamlit run app.py进入前端界面
import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 0. 配置路径 (必须与 build_db.py 保持一致) ---
DB_PATH = "./chroma_db_pro"

# --- 1. 页面基础设置 ---
st.set_page_config(page_title="材料知识库Pro", layout="wide")
st.title("🧪 材料科学知识库系统 (Pro版)")
st.caption("🚀 全库检索模式 | 数据已持久化存储")
st.markdown("---")

# --- 2. 侧边栏：状态监控与配置 ---
with st.sidebar:
    st.header("系统状态 monitor")

    # 检查数据库是否存在
    if os.path.exists(DB_PATH):
        st.success(f"✅ 本地知识库已连接\n\n路径: `{DB_PATH}`")
    else:
        st.error("❌ 未找到数据库！")
        st.warning("请先运行 `build_db.py` 构建知识库。")

    st.markdown("---")
    st.header("模型配置")
    # 把列表里的 "llama3.2" 换成 "qwen2.5:7b"
    selected_model = st.selectbox("选择本地模型", ["qwen2.5:7b"], index=0)

    # 添加一个清除历史的按钮
    if st.button("🗑️ 清空聊天记录"):
        st.session_state.messages = []
        st.rerun()


# --- 3. 核心函数 (加载资源) ---
@st.cache_resource
def get_vector_db():
    """直接从硬盘加载已经建好的向量数据库"""
    print(f"正在加载数据库: {DB_PATH}")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # persist_directory 指向之前 build_db.py 生成的文件夹
    vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    return vector_db

#默认的 temperature (温度) 可能偏高，导致模型想“发挥创意”。做科研问答，我们需要它像机器人一样死板。
#把温度降到 0，强制它完全基于事实。
@st.cache_resource
def get_llm(model_name):
    # 将 temperature 改为 0 (最严谨模式)
    return OllamaLLM(model=model_name, temperature=0)


# --- 4. 主界面逻辑 ---

# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 检查数据库是否就绪
if not os.path.exists(DB_PATH):
    st.info("👋 欢迎使用！请先在 PyCharm 中运行 `build_db.py` 来构建你的材料数据库，然后刷新本页面。")
    st.stop()  # 停止往下执行

# 加载数据库 (利用缓存，只会加载一次)
vector_db = get_vector_db()

# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 5. 处理用户提问 ---
if prompt := st.chat_input("请输入关于文献的问题 (将检索 data 目录下的所有 PDF)..."):
    # 1. 显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    # 2. AI 回答
    with st.chat_message("assistant"):
        msg_placeholder = st.empty()
        msg_placeholder.markdown("🔍 正在全库检索...")

        try:
            # A. 检索器：查找最相关的 4 个片段
            retriever = vector_db.as_retriever(search_kwargs={"k": 10})
            llm = get_llm(selected_model)

            # B. 提示词模板 (优化版)
            # --- 针对 PDF 乱码优化的 Prompt ---
            template = """
            你是一个材料科学专家。请根据下方的【参考资料】回答【用户问题】。

            ⚠️以此为准：
            1. **自动纠错**：参考资料可能包含 PDF 识别错误（例如单词粘连 "Wintercalates" -> "W intercalates"），请尝试理解其真实含义。
            2. **提取事实**：重点寻找与问题相关的**化学式、数字、位置关系**。
            3. **语言要求**：尽量用中文回答，但保留英文专有名词。
            4. 如果实在找不到任何相关线索，再说“未找到”。
            5. 参考资料保留了原始 PDF 的**视觉布局**。
            6. 表格是通过**空格和换行**对齐的。

            【参考资料】：
            {context}

            【用户问题】：
            {question}
            """
            rag_prompt = ChatPromptTemplate.from_template(template)


            # C. 构建 RAG 流水线
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)


            rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | rag_prompt
                    | llm
                    | StrOutputParser()
            )

            # D. 执行检索与生成
            # 先检索一遍，为了在界面上展示来源 (Debug用)
            retrieved_docs = retriever.invoke(prompt)

            # 生成回答
            response = rag_chain.invoke(prompt)
            msg_placeholder.markdown(response)

            # 保存回答到历史
            st.session_state.messages.append({"role": "assistant", "content": response})

            # E. 核心亮点：展示检索来源 (论文加分项)
            with st.expander("📚 查看来源文档 (Evidence)"):
                for i, doc in enumerate(retrieved_docs):
                    # 获取文件名 (source metadata)
                    source_path = doc.metadata.get("source", "未知来源")
                    file_name = os.path.basename(source_path)  # 只显示文件名，不显示长路径

                    st.markdown(f"**来源 {i + 1}:** `{file_name}`")
                    st.caption(f"内容摘要: {doc.page_content[:500]}...")  # 只显示前500字
                    st.markdown("---")

        except Exception as e:
            st.error(f"发生错误: {e}")