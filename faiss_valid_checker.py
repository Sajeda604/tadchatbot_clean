import os
import sys

import faiss  # type: ignore


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
INDEX_PATH = os.path.join(DB_FAISS_PATH, "index.faiss")


def main() -> int:
    """يتأكد أن ملف index.faiss موجود وقابل للقراءة من FAISS."""

    if not os.path.isdir(DB_FAISS_PATH):
        print(f"❌ لم يتم العثور على مجلد قاعدة البيانات: {DB_FAISS_PATH}")
        return 1

    if not os.path.isfile(INDEX_PATH):
        print(f"❌ لم يتم العثور على ملف index.faiss في: {INDEX_PATH}")
        print("➡️ شغّل: python faiss_build.py ثم أعد محاولة التحقق.")
        return 1

    try:
        _ = faiss.read_index(INDEX_PATH)
    except Exception as exc:  # pragma: no cover - تشخيص فقط
        print(f"❌ تعذر قراءة index.faiss: {exc}")
        return 1

    print("✅ index.faiss قابل للقراءة وجاهز للاستخدام.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
