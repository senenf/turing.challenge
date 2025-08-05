import gradio as gr
import os
import json
import uuid
import shutil

from dotenv import load_dotenv
from datetime import datetime

from chatbot.chat import chat_with_agent
from chatbot.memory import get_memory

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from pipeline.loader import load_pdf_chunks
from pipeline.indexer import index_documents
from pipeline.image_processor import index_image_documents, extract_images_from_pdf


# Load environment variables
load_dotenv("utils/.env")

# Ensure folders exist and DB is initialized
def setup_environment():
    # Create folders if they don't exist
    os.makedirs("../src/sample_data", exist_ok=True)
    os.makedirs("../src/db", exist_ok=True)

    # Check if the DB already exists by looking for Chroma DB index
    chroma_index_file = os.path.join("../src/db", "chroma-collections.parquet")
    if not os.path.exists(chroma_index_file):
        print("🛠️ Initializing Chroma vector database...")

        # Load env vars
        load_dotenv("utils/.env")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        embedding_json = os.getenv("EMBEDDING_MODELS_JSON")
        if not embedding_json:
            raise EnvironmentError("Missing EMBEDDING_MODELS_JSON in environment variables")

        embedding_models = json.loads(embedding_json)
        EMBEDDING_MODEL = embedding_models.get("small", os.getenv("DEFAULT_EMBEDDING_MODEL"))

        # Setup embedding
        embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, model=EMBEDDING_MODEL)

        # Create and persist Chroma DB
        db = Chroma(
            collection_name="knowledge_base",
            embedding_function=embedding,
            persist_directory="../src/db"
        )
        print("✅ Chroma vector DB initialized.")
    else:
        print("📦 Chroma DB already initialized.")

setup_environment()

# Load PDFs
def get_available_pdfs():
    return [f for f in os.listdir("sample_data") if f.endswith(".pdf")]

pdfs = get_available_pdfs()

# In-memory conversation storage
conversations = {}

def create_or_get_conversation(name=None):
    if not name:
        name = f"convo-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    if name not in conversations:
        conversations[name] = {"history": [], "docs": [], "memory": get_memory()}
    return name, conversations[name]

def respond(message, convo_name):
    auto_created = False
    if not convo_name:
        convo_name = f"convo-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        _, convo = create_or_get_conversation(convo_name)
        auto_created = True
    else:
        convo = conversations[convo_name]
        if "memory" not in convo:
            convo["memory"] = get_memory()

    answer = chat_with_agent(message, convo["docs"], convo["memory"])
    convo["history"].append({"role": "user", "content": message})
    convo["history"].append({"role": "assistant", "content": answer})

    
    return (
        convo["history"],
        convo_name,
        gr.update(choices=list(conversations.keys()), value=convo_name),
        "",  # limpiar input
        convo_name if auto_created else gr.update()  # si se creó, actualizamos la caja
    )

def load_conversation(name):
    convo_name, convo = create_or_get_conversation(name)
    return convo_name, convo["history"], convo["docs"], gr.update(interactive=True)

def update_documents(convo_name, selected):
    if convo_name in conversations:
        conversations[convo_name]["docs"] = selected
        print(f"Selected docs for {convo_name}: {selected}")



def add_conversation(new_name):
    if not new_name:
        new_name = f"convo-{uuid.uuid4().hex[:8]}"
    if new_name not in conversations:
        conversations[new_name] = {"history": [], "docs": [], "memory": get_memory()}
    return (
        gr.update(choices=list(conversations.keys()), value=new_name),
        new_name,
        [],
        [],
        gr.update(interactive=True),
        ""
    )

def upload_and_index_pdf(uploaded_file):
    if not uploaded_file:
        return gr.update(), pdfs, gr.update(value=None)

    # Save PDF to ../sample_data/
    temp_path = uploaded_file.name
    filename = os.path.basename(temp_path)
    save_path = os.path.join("../sample_data", filename)

    # ✅ Copy the temp file to your storage
    shutil.copyfile(temp_path, save_path)

    # ✅ Index text
    try:
        chunks = load_pdf_chunks(save_path)
        index_documents(chunks, persist_directory="db", collection_name="knowledge_base")
    except Exception as e:
        print(f"⚠️ Failed to index text: {e}")

    # ✅ Extract and index images
    try:
        extract_images_from_pdf(save_path)  # saves to _extracted_images folder
        image_folder = save_path.replace(".pdf", "_extracted_images")
        if os.path.exists(image_folder):
            index_image_documents(image_folder)
    except Exception as e:
        print(f"⚠️ Failed to index images: {e}")

    # ✅ Force persist and reload Chroma
    try:

        vectordb = Chroma(
            collection_name="knowledge_base",
            persist_directory="db",
            embedding_function=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        )
        vectordb.persist()  # force write
        print("✅ Chroma DB persisted successfully.")
    except Exception as e:
        print(f"⚠️ Error persisting vector DB: {e}")

    
    # ✅ Update file list
    new_pdfs = [f for f in os.listdir("../sample_data") if f.endswith(".pdf")]
    return gr.update(choices=new_pdfs, value=filename), new_pdfs, gr.update(value=None)


#-------------------------------------------------#

# Interface
with gr.Blocks(css="""
#chatbox .gr-chatbot {
    margin: 0 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border-radius: 12px;
    background-color: #fafafa;
}
.gr-message.user {
    background: #e1f0ff !important;
    color: #000;
}
.gr-message.assistant {
    background: #f1f1f1 !important;
    color: #000;
}
""") as app:
    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### 💬 Conversaciones")
            convo_selector = gr.Dropdown(choices=list(conversations.keys()), label="Seleccionar", interactive=True)
            new_convo_name = gr.Textbox(placeholder="Nombre de conversación (opcional)", label="")
            create_btn = gr.Button("➕ Nueva conversación")
            gr.Markdown("### 📄 Documentos")
            file_selector = gr.CheckboxGroup(choices=pdfs, label="Usar en esta conversación")

            gr.Markdown("### 📤 Subir nuevo PDF")
            file_upload = gr.File(label="Sube un archivo PDF", file_types=[".pdf"])
            upload_btn = gr.Button("📥 Indexar documento")
            
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(label="", height=500, type="messages", elem_id="chatbox")
            msg = gr.Textbox(placeholder="Escribe tu mensaje...", show_label=False, interactive=True)

    selected_convo = gr.State(value="")

    # Crear conversación (botón o enter)
    for trigger in [create_btn.click, new_convo_name.submit]:
        trigger(
            add_conversation,
            inputs=new_convo_name,
            outputs=[convo_selector, selected_convo, chatbot, file_selector, msg, new_convo_name]
        )

    convo_selector.change(
        load_conversation,
        inputs=convo_selector,
        outputs=[selected_convo, chatbot, file_selector, msg]
    )

    file_selector.change(
        update_documents,
        inputs=[selected_convo, file_selector],
        outputs=[]
    )

    # Enviar mensaje y bloquear solo la caja de texto
    msg.submit(
        fn=respond,
        inputs=[msg, selected_convo],
        outputs=[chatbot, selected_convo, convo_selector, msg, new_convo_name],
        api_name="submit"
    ).then(
        lambda: gr.update(interactive=True),  # reactivar input después
        None,
        [msg]
    )
    upload_btn.click(
    fn=upload_and_index_pdf,
    inputs=[file_upload],
    outputs=[file_selector, gr.State(), file_upload]
)



app.launch()
