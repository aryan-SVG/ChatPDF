import gradio as gr, os, warnings, shutil, uuid
from typing import List, Dict, Any
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


class PDFProcessor:
    """Loads → splits → embeds → stores → builds a retrieval-QA chain."""
    def __init__(self) -> None:
        self.vector_store: Chroma | None = None
        self.qa_chain: ConversationalRetrievalChain | None = None
        self.processed: list[str] = []
        self.persist_dir = f"/tmp/chroma_{uuid.uuid4().hex}"
        os.makedirs(self.persist_dir, exist_ok=True)

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
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = splitter.split_documents(docs)

            # 3) embed & store in memory (no persistence to disk!)
            embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(chunks, embedder)
            else:
                self.vector_store.add_documents(chunks)

            # 4) build / refresh conversational retrieval chain
            llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_retries=2)
            retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 15})
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=memory,
                return_source_documents=True,
                output_key="answer"
            )

            self.processed += [f.name for f in new_files]
            return "PDFs processed."
        except Exception as e:
            return f"Error while processing PDFs: {e}"

    def query_initial(self, business: str, domain: str, prospects: str) -> str:
        """The one-time, full-context prompt after the three questions."""
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
3. **Sort** them in *strict* descending likelihood:  
   - All “High” matches first (in order),  
   - then “Medium”,  
   - then “Low”.  
   *Do not* interleave—High must come before any Medium, etc.    
4. For each person, print a mini visiting card exactly like:

────────────────────────
Name        : full name
Role        : title + company
Contact     : email or phone or 'N/A'
Website     : URL or Company/personal site or 'N/A'
Address     : City + state or full address if present, or "N/A"
Why relevant: 1-line reason
Likelihood to help: High / Medium / Low  

────────────────────────

If no such people exist, reply _exactly_: **No relevant matches found in the PDF.**
"""
        try:
            return self.qa_chain.invoke({"question": prompt})["answer"]
        except Exception as e:
            return f"Error during initial query: {e}"

    def ask_followup(self, question: str) -> str:
        """Use the same chain & memory for follow-up questions."""
        if not self.qa_chain:
            return "Please upload and process a PDF first."
        try:
            return self.qa_chain.invoke({"question": question})["answer"]
        except Exception as e:
            return f"Error during follow-up query: {e}"


def build_ui() -> gr.Blocks:
    proc = PDFProcessor()
    init_state: Dict[str, Any] = {
        "step": 0,
        "business": "",
        "domain": "",
        "prospects": "",
        "pdf_ok": False
    }

    def on_process(files, st):
        msg = proc.process_pdfs(files)
        st["pdf_ok"] = msg.startswith("PDFs processed")
        st["step"] = 0
        bot_msg = ("System", "✅ PDFs processed. Let's begin.\n\nWhat business are you in?")
        # clear textbox
        return gr.update(value=""), [bot_msg], st

    def chat_logic(user, history, st):
        # If PDFs not yet processed
        if not st["pdf_ok"]:
            history.append((user, "Please upload a PDF and click **Process PDFs** first."))
            return history, st

        # Step 0: ask business
        if st["step"] == 0:
            st["business"] = user
            st["step"] = 1
            history.append((user, "In which domain do you need help?"))
        # Step 1: ask domain
        elif st["step"] == 1:
            st["domain"] = user
            st["step"] = 2
            history.append((user, "Who are your ideal clients or prospects?"))
        # Step 2: ask prospects
        elif st["step"] == 2:
            st["prospects"] = user
            st["step"] = 3
            # now fire initial full-context prompt
            answer = proc.query_initial(st["business"], st["domain"], st["prospects"])
            history.append((user, answer))
        # Step ≥3: any follow-up => use memory
        else:
            answer = proc.ask_followup(user)
            history.append((user, answer))

        return history, st

    with gr.Blocks(title="Photosheet Business Referral Chatbot", css="assets/style.css") as demo:
        
        with gr.Row(): 
          # ←CHANGE .css
            gr.Image(
                "assets/sociosquare_logo.png",
                show_label=False,
                show_download_button=False,
                interactive=False,     
                height=28


            )
            

        #  Privacy Notice 
        # replace any inline <style> block with:
        gr.HTML(
            "<div class='privacy'>"
            "🔒 Your data remains private, will be deleted after a few days, "
            "and will never train the LLM."
            "</div>"
           )


        gr.Markdown("# 📄 Photosheet Business Referral Chatbot")

        with gr.Row():
            file_box = gr.File(label="Upload PDF", file_count="multiple", file_types=[".pdf"])
            proc_btn = gr.Button("Process PDFs")

        chat_box = gr.Chatbot(
            value=[("System", "👋 Please upload your photosheet PDF and click **Process PDFs** to begin.")],
            label="Referral Assistant"
        )
        txt_in = gr.Textbox(placeholder="Type here and press Enter…")
        state = gr.State(init_state)

        proc_btn.click(on_process, inputs=[file_box, state], outputs=[txt_in, chat_box, state])
        txt_in.submit(chat_logic, inputs=[txt_in, chat_box, state], outputs=[chat_box, state]).then(
            lambda: gr.update(value=""), None, txt_in
        )

    return demo

if __name__ == "__main__":
    build_ui().launch(share=False)

