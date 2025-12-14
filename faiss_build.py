import os
import sys

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")


def build_faiss_index() -> None:
    """بناء قاعدة FAISS من الملفات الموجودة في مجلد data/ (.txt و .pdf)."""

    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise RuntimeError("HF_TOKEN غير مضبوط في متغيرات البيئة. قم بضبطه قبل بناء قاعدة FAISS.")

    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"مجلد البيانات غير موجود: {DATA_DIR}")

    docs = []

    # تحميل ملفات النصوص و PDF
    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.lower().endswith((".txt", ".pdf")):
            continue

        file_path = os.path.join(DATA_DIR, filename)
        print(f"📄 تحميل الملف: {filename}")

        try:
            if filename.lower().endswith(".txt"):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                loader = PyPDFLoader(file_path)

            docs.extend(loader.load())
        except Exception as e:  # pragma: no cover - فقط للرسائل التشخيصية
            print(f"⚠️ تعذر تحميل الملف {filename}: {e}")

    if not docs:
        print("❌ لم يتم العثور على أي مستندات صالحة في مجلد data/.")
        return

    print("✂️ تقسيم المستندات إلى مقاطع...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    texts = text_splitter.split_documents(docs)

    print("🧠 إنشاء التضمينات باستخدام HuggingFaceEndpointEmbeddings...")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token,
    )

    print("⚙️ جاري إنشاء قاعدة بيانات FAISS...")
    db = FAISS.from_documents(texts, embeddings)

    os.makedirs(DB_FAISS_PATH, exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ تم إنشاء قاعدة بيانات FAISS وحفظها في: {DB_FAISS_PATH}")


def main() -> None:
    try:
        build_faiss_index()
    except Exception as exc:  # pragma: no cover - سلوك تفاعلي فقط
        print(f"❌ حدث خطأ أثناء بناء قاعدة FAISS: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
