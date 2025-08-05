from langchain.agents import initialize_agent, AgentType, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from .memory import get_memory
from config import OPENAI_API_KEY, MODEL_NAME
import os

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K"))  # Number of documents to retrieve

# 1. Chroma retriever with filtering by selected files
def make_rag_tool(selected_files=None, k=RETRIEVAL_K):
    vectordb = Chroma(
        collection_name="knowledge_base",
        embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        persist_directory="db"
    )

    #IGNORE
    # 🧼 Normalize selected file names
    # normalized_files = None
    # if selected_files:
    #     normalized_files = [os.path.basename(f).strip().lower() for f in selected_files]

    def retrieval_tool_fn(query: str) -> str:
        filter_conditions = []

        # ✅ Add file filter if provided
        if selected_files:
            filter_conditions.append({"source": {"$in": selected_files}})

        # ✅ Add type=“image” if query suggests visual intent
        image_keywords = ["image", "photo", "picture", "diagram", "visual", "show", "see", "see in the image"]
        if any(kw in query.lower() for kw in image_keywords):
            filter_conditions.append({"type": "image"})

        # ✅ Compose final search_kwargs
        search_kwargs = {"k": k}
        if len(filter_conditions) == 1:
            search_kwargs["filter"] = filter_conditions[0]
        elif len(filter_conditions) > 1:
            search_kwargs["filter"] = {"$and": filter_conditions}

        retriever = vectordb.as_retriever(search_kwargs=search_kwargs)
        docs = retriever.get_relevant_documents(query)

        if not docs:
            return "❌ No relevant documents or images found."

        # ✅ Format results
        formatted = []
        for d in docs:
            meta = d.metadata
            if meta.get("type") == "image":
                formatted.append(
                    f"[📷 Imagen]\n"
                    f"📝 Descripción: {d.page_content}\n"
                    f"📄 PDF: {meta.get('source', 'N/A')}\n"
                    f"🖼️ Imagen: {meta.get('image_file', 'N/A')}\n"
                    f"📄 Página: {meta.get('page_number', 'N/A')}"
                    f"(Para referencia: Archivo de imagen: {meta.get('image_file', 'N/A')} del PDF {meta.get('source', 'N/A')})"
                )
            else:
                formatted.append(
                    f"[📄 Texto]\n"
                    f"{d.page_content}\n"
                    f"📄 PDF: {meta.get('source', 'N/A')}, Página: {meta.get('page_number', 'N/A')}"
                )

        return "\n\n".join(formatted)

    return Tool(
        name="document_retriever",
        description="Useful for answering questions using indexed documents and images. Input can ask about content, images, captions, or specific PDFs.",
        func=retrieval_tool_fn
    )



# 2. Create agent using tools
def get_agent(selected_files=None, memory=None):
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

    tools = [PythonREPLTool()] ##code execution tool

    rag_tool = make_rag_tool(selected_files, k=RETRIEVAL_K) ## Document retrieval tool

    if rag_tool:
        tools.append(rag_tool)

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )


# 3. External callable from Gradio
def chat_with_agent(message, selected_files=None, memory=None):
    agent = get_agent(selected_files, memory)
    result = agent.invoke({"input": message})
    try:
        return result["output"] if isinstance(result, dict) and "output" in result else result
    except Exception as e:
        if isinstance(result, dict) and "output" in result:
            return str(result["output"])
        return str(result)

