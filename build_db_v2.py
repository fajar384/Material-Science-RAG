import os
import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- 配置路径 ---
DATA_PATH = "./data"
DB_PATH = "./chroma_db_pro"


def load_pdf_visual_layout(pdf_path):
    print(f"📖 正在按【视觉布局】解析: {os.path.basename(pdf_path)} ...")
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # ✅ 核心：使用 layout=True 保留表格的物理对齐格式
            # 这样 AI 就能像人眼一样，通过空格位置看出谁对应谁
            text = page.extract_text(layout=True) or ""

            metadata = {"source": pdf_path, "page": i + 1}
            documents.append(Document(page_content=text, metadata=metadata))

    return documents


def create_vector_db_pro():
    pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")]
    if not pdf_files:
        print("❌ data 文件夹是空的！")
        return

    all_docs = []
    for pdf_file in pdf_files:
        path = os.path.join(DATA_PATH, pdf_file)
        docs = load_pdf_visual_layout(path)
        all_docs.extend(docs)

    print(f"✅ 解析完成，共 {len(all_docs)} 页。")

    # 确保整整一页（甚至两页）都在同一个片段里，绝不切断表格。
    # --- 🔴 核心修改点：改回小窗口 ---
    # 这样表格的每一行都会变成一个独立的、高权重的片段
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=["\n", " ", ""]  # 优先按行切，保护表格行
    )

    splits = text_splitter.split_documents(all_docs)
    print(f"✂️ 共切分为 {len(splits)} 个片段 (数量应该会变多)")
    print("🧠 正在重建数据库...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print(f"🎉 视觉布局版数据库构建成功！已保存到 {DB_PATH}")


if __name__ == "__main__":
    create_vector_db_pro()