import gradio as gr, os, warnings, shutil, uuid
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ────────────────────────── env / housekeeping ──────────────────────────
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# ─────────────────────────── helper class ───────────────────────────────
class PDFProcessor:
    """Loads → splits → embeds → stores → builds a retrieval-QA chain."""
    def __init__(self) -> None:
        self.vector_store: Chroma | None = None
        self.qa_chain:     ConversationalRetrievalChain | None = None
        self.processed:    list[str] = []
        # keep every run isolated → random folder inside /tmp
        self.persist_dir = f"/tmp/chroma_{uuid.uuid4().hex}"
        os.makedirs(self.persist_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────
    def process_pdfs(self, pdf_files: List[gr.File]) -> str:
        try:
            new_files = [f for f in pdf_files if f.name not in self.processed]
            if not new_files:
                return "No new PDFs to process."

            # 1) load
            docs = []
            for f in new_files:
                docs.extend(PyPDFLoader(f.name).load())

            # 2) split
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""])
            chunks = splitter.split_documents(docs)

            # 3) embed & store
            embedder = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2")
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    chunks, embedder, persist_directory=self.persist_dir)
            else:
                self.vector_store.add_documents(chunks)
            self.vector_store.persist()

            # 4) build / refresh chain
            llm = ChatGroq(model="llama-3.3-70b-versatile",
                           temperature=0, max_retries=2)
            retriever = self.vector_store.as_retriever(
                search_type="mmr",   # more diverse results
                search_kwargs={"k": 15})
            memory = ConversationBufferMemory(memory_key="chat_history",
                                              return_messages=True,
                                              output_key="answer")
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory,
                return_source_documents=True, output_key="answer")

            self.processed += [f.name for f in new_files]
            return "PDFs processed."
        except Exception as e:      # debug-friendly
            return f"Error while processing PDFs: {e}"

    # ────────────────────────────────────────────────────────────────────
    def query(self, business: str, domain: str, prospects: str) -> str:
        if not self.qa_chain:
            return "Please upload and process a PDF first."

        prompt = f"""
You are an assistant helping users find potential business partners 
from the uploaded PDF photosheet.

 USER CONTEXT 
The user is in the business of: {business}
The user needs help in the domain of: {domain}
The user's ideal clients or prospects are: {prospects}


**Task**

1. Examine the retrieved PDF snippets.  
2. List **every person who could even remotely help** with *{domain}*  
   (Either they work in it or build tech for it or they build tech similar  or can intro someone). 
3. Sort the people **from most to least likely to help**. 
4. For each person, print a mini visiting card exactly like:

────────────────────────
Name        : <full name>
Role        : <title + company>
Contact     : <email or phone or 'N/A'>
Website     : <URL or Company/personal site or 'N/A'>
Address     : <City + state or full address if present, or "N/A">
Why relevant: <1-line reason>
────────────────────────

If no such people exist, reply _exactly_: **No relevant matches found in the PDF.** 
"""
        try:
            ans = self.qa_chain.invoke({"question": prompt})["answer"]
            return ans
        except Exception as e:
            return f"Error during query: {e}"

# ─────────────────────────── UI wiring ──────────────────────────────────
def build_ui() -> gr.Blocks:
    proc = PDFProcessor()
    INIT_STATE: Dict[str, Any] = {"step": 0, "business": "", "domain": "",
                                  "prospects": "", "pdf_ok": False}

    # ---------- callbacks ----------
    def on_process(files, st):
        msg = proc.process_pdfs(files)
        st["pdf_ok"] = msg.startswith("PDFs processed")
        st["step"] = 0
        bot_seed = ("System",
                    "✅ PDFs processed. Let's begin.\n\nWhat business are you in?")
        return gr.update(value=""), [bot_seed], st

    def chat(user, hist, st):
        if not st["pdf_ok"]:
            hist.append((user, "Please upload a PDF and click **Process PDFs** first."))
            return hist, st

        if st["step"] == 0:
            st["business"] = user; st["step"] = 1
            hist.append((user, "In which domain do you need help?"))
        elif st["step"] == 1:
            st["domain"] = user; st["step"] = 2
            hist.append((user, "Who are your ideal clients or prospects?"))
        elif st["step"] == 2:
            st["prospects"] = user; st["step"] = 3
            answer = proc.query(st["business"], st["domain"], st["prospects"])
            hist.append((user, answer))
        else:  # follow-ups reuse context
            answer = proc.query(st["business"], st["domain"], st["prospects"])
            hist.append((user, answer))
        return hist, st

    # ---------- layout ----------
    with gr.Blocks(title="Provisor Business Referral Chatbot") as demo:
        gr.Markdown("# 📄 Provisor Business Referral Chatbot")

        with gr.Row():
            file_box   = gr.File(label="Upload PDF", file_count="multiple",
                                 file_types=[".pdf"])
            proc_btn   = gr.Button("Process PDFs")

        chat_box = gr.Chatbot(
            value=[("System", "👋 Please upload your photosheet PDF and "
                              "click **Process PDFs** to begin.")],
            label="Referral Assistant")
        txt_in   = gr.Textbox(placeholder="Type message and press Enter…")
        state    = gr.State(INIT_STATE)

        # hook up events
        proc_btn.click(on_process,
                       inputs=[file_box, state],
                       outputs=[txt_in, chat_box, state])
        txt_in.submit(chat,
                      inputs=[txt_in, chat_box, state],
                      outputs=[chat_box, state]).then(
            lambda: gr.update(value=""), None, txt_in)

    return demo

if __name__ == "__main__":
    build_ui().launch(share=False)
