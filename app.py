import os
import traceback
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate

# تحميل متغيرات البيئة
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
# تعديل RTL للغة العربية
st.markdown(
    """
    <style>
    html, body, .main {
        direction: rtl;
        text-align: right;
    }
    .st-chat-message > div {
        direction: rtl;
        text-align: right;
    }
    .stTextInput>div>input {
        direction: rtl;
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# تحميل قاعدة FAISS
@st.cache_resource
def get_vectorstore():
    """تحميل قاعدة FAISS من المسار المحدد مع التحقق من وجود الملفات."""

    index_path = os.path.join(DB_FAISS_PATH, "index.faiss")
    meta_path = os.path.join(DB_FAISS_PATH, "index.pkl")

    if not os.path.isdir(DB_FAISS_PATH) or not (
        os.path.isfile(index_path) and os.path.isfile(meta_path)
    ):
        raise FileNotFoundError(
            "لم يتم العثور على قاعدة FAISS. "
            "شغّلي faiss_build.py محليًا ثم ارفعي مجلد vectorstore/db_faiss إلى المستودع "
            "أو شغّل CI لبنائها تلقائيًا."
        )

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=HF_TOKEN,
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    return db

# قائمة التحيات
GREETINGS = ["مرحبا", "أهلاً", "أهلا", "هاي", "السلام عليكم"]

# قائمة الأسئلة المقترحة بناءً على البيانات
SUGGESTED_QUESTIONS = [
    "ما هي خدمات تطوير التطبيقات التي يقدمها المكتب؟",
    "هل يقدم المكتب حلول الذكاء الاصطناعي والـ Chatbots؟",
    "ما هي أنظمة ERP و CRM التي يطورها المكتب؟",
    "كيف يقوم المكتب بتطوير المتاجر الإلكترونية؟",
    "ما هي خدمات الأمن السيبراني المتاحة؟",
    "هل يوفر المكتب خدمات استضافة وسيرفرات؟",
    "ما هي المشاريع السابقة للمكتب؟",
    "كيف يتم تطوير نظام حجز المواعيد الطبي؟",
    "ما هي تقنيات التعلم الآلي التي يستخدمها المكتب؟",
    "هل يقدم المكتب خدمات التسويق الرقمي؟",
    "ما هي القيم الأساسية للمكتب؟",
    "كيف يتم بناء منصة تعليم إلكترونية؟",
]

# Prompt للبوت الذكي + سؤال متابعة
retrieval_prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
أنت مساعد ذكي متخصص في الإجابة عن أسئلة حول خدمات المكتب البرمجي. أنت دقيق، منطقي، وقادر على الفهم والتفسير. مهمتك:

1. إذا كان السؤال موجودًا في السياق (context)، أجب بطريقة واضحة ومقنعة، حتى لو كان السؤال مختصرًا أو استخدم كلمات مشابهة.
2. إذا لم يكن السؤال موجودًا نصيًا في المستندات، حاول شرح الإجابة بناءً على المعرفة العامة عن صناعة البرمجيات، مع الإشارة إلى أنها من المعرفة العامة وليست من المستندات.
3. اجعل إجابتك مفهومة وسهلة، قصيرة وواضحة ومقنعة.
4. لا تختلق معلومات غير صحيحة.
5. بعد كل إجابة، اقترح سؤال متابعة قصير وملائم يتعلق بخدمات المكتب.
6. إذا لم تعرف الإجابة، أجب: "لا توجد معلومات متاحة في المستندات حول هذا الموضوع".

السياق (المستندات):
{context}

سؤال المستخدم:
{input}

إجابة (مع اقتراح سؤال متابعة في النهاية):
"""
)

def main():
    st.title("🤖 مساعد المكتب البرمجي الذكي")
    
    # عرض الأسئلة المقترحة في الشريط الجانبي
    with st.sidebar:
        st.markdown("### 💡 أسئلة مقترحة")
        st.markdown("اختر أحد الأسئلة أدناه أو اكتب سؤالك الخاص:")
        
        selected_question = st.selectbox(
            "اختر سؤال:",
            options=[""] + SUGGESTED_QUESTIONS,
            label_visibility="collapsed"
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # عرض المحادثات السابقة
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # إذا تم اختيار سؤال من القائمة، استخدمه
    if selected_question:
        user_prompt = selected_question
    else:
        user_prompt = st.chat_input("اكتب سؤالك أو تحيتك هنا...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        
        if "صباح الخير" in user_prompt:
            greeting_reply = "صباح النور! كيف يمكنني مساعدتك اليوم؟"
        elif "مساء الخير" in user_prompt:
            greeting_reply = "مساء النور! كيف يمكنني مساعدتك اليوم؟"
        elif any(greet in user_prompt for greet in GREETINGS):
            greeting_reply = "مرحبا! كيف يمكنني مساعدتك اليوم؟"
        else:
            greeting_reply = None

        if greeting_reply:
            st.chat_message("assistant").markdown(greeting_reply)
            st.session_state.messages.append({"role": "assistant", "content": greeting_reply})
        else:
            try:
                # التحقق من متغيرات البيئة الأساسية قبل استدعاء النماذج
                if not HF_TOKEN:
                    st.error("متغير البيئة HF_TOKEN غير مضبوط. يرجى ضبطه قبل تشغيل التطبيق.")
                    return
                if not GROQ_API_KEY:
                    st.error("متغير البيئة GROQ_API_KEY غير مضبوط. يرجى ضبطه قبل تشغيل التطبيق.")
                    return

                # محاولة تحميل قاعدة FAISS مع رسالة واضحة عند عدم توفرها
                try:
                    db = get_vectorstore()
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.info(
                        "لتوليد قاعدة FAISS محليًا نفّذ: python faiss_build.py "
                        "ثم ارفع مجلد vectorstore/db_faiss إلى GitHub/Railway."
                    )
                    return

                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.0,
                    max_tokens=512,
                    api_key=GROQ_API_KEY
                )

                # إنشاء سلسلة RAG مع البحث الموسع
                combine_chain = create_stuff_documents_chain(llm, retrieval_prompt)
                rag_chain = create_retrieval_chain(db.as_retriever(search_kwargs={"k": 10}), combine_chain)

                # استدعاء RAG
                result = rag_chain.invoke({"input": user_prompt})

                # استخراج النص الفعلي فقط
                if isinstance(result, dict):
                    answer = result.get("output_text") or result.get("text")
                elif hasattr(result, "content"):
                    answer = result.content
                else:
                    answer = str(result)

                # إذا لم توجد نتيجة من المستندات → نستخدم LLM للمعرفة العامة
                if not answer or "لا توجد معلومات متاحة" in answer:
                    llm_response = llm.invoke(
                        f"أجب على السؤال التالي حول الخدمات البرمجية بطريقة واضحة ومبسطة للمستخدم العادي. "
                        f"إذا لم تعرف الإجابة بدقة، أجب 'لا توجد معلومات متاحة': {user_prompt}"
                    )
                    if hasattr(llm_response, "content"):
                        answer = llm_response.content
                    else:
                        answer = str(llm_response)
                    answer = f"ملاحظة: هذه المعلومات من المعرفة العامة. {answer}"

                st.chat_message("assistant").markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"حدث خطأ داخلي: {repr(e)}")
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

