import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")


def build_rag_chain():
    """تهيئة سلسلة RAG لاختبار قاعدة FAISS من سطر الأوامر."""

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN غير مضبوط في متغيرات البيئة.")
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY غير مضبوط في متغيرات البيئة.")

    if not os.path.isdir(DB_FAISS_PATH):
        raise FileNotFoundError(
            f"لم يتم العثور على قاعدة FAISS في المسار: {DB_FAISS_PATH}. "
            "شغّل faiss_build.py أولاً لإنشائها."
        )

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=HF_TOKEN,
    )

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.1,
        max_tokens=512,
        api_key=GROQ_API_KEY,
    )

    prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 5}), combine_chain)
    return rag_chain


def main():
    try:
        rag_chain = build_rag_chain()
    except Exception as e:
        print(f"حدث خطأ أثناء تهيئة سلسلة RAG: {e}")
        return

    user_query = input("💬 اكتب سؤالك حول الخدمات البرمجية هنا: ")
    if not user_query:
        print("لم يتم إدخال أي سؤال.")
        return

    response = rag_chain.invoke({"input": user_query})

    answer = None
    if isinstance(response, dict):
        answer = (
            response.get("answer")
            or response.get("output_text")
            or response.get("text")
        )
    else:
        answer = str(response)

    print("\n🤖 الإجابة:")
    if answer is not None:
        print(answer)
    else:
        print("لم يتم استرجاع إجابة من السلسلة.")

    if isinstance(response, dict) and response.get("context"):
        print("\n📚 المستندات المصدرية:")
        for doc in response["context"]:
            print(f"- {doc.metadata} -> {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
