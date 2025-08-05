# Turing Challenge 

This repository contains the answer to the requested challenges. Code can be found in each of the folders.

Answers are provided to justify each choice of technology/stack.

## Getting Started

Navigate to the repository from the command line and execute the following command to create the environment:

```bash
conda env create -f environment.yaml
```

Now visualize it and activate it in the command line:

```bash
conda env list
conda activate turing-challenge
```

## 1. [Chatbot Overview](./1.chatbot/)

### Interface

- **Gradio**: Fast prototyping and testing of chat features.  
- **FastAPI + Vite + React (TypeScript)**: Ideal for production environments _(not covered here)_.

**Features Implemented:**
- Multi-conversation support
- Chat about selected documents or the entire vector database when none selected.
- Document upload, indexing, and integration with chat.

---

### Vector Database

#### Production-Ready Options

**AWS**  
- Amazon OpenSearch  
- Aurora (PostgreSQL) with `pgvector`  
- S3-based vector storage _(experimental)_

**Google Cloud (GCP)**  
- Vertex AI + Vector Search  
- AlloyDB + `pgvector`

**Microsoft Azure**  
- Azure Cognitive Search + Vector Search  
- PostgreSQL + `pgvector`

> _Note: Currently skipping Databricks, Oracle, and Redis._

---

#### Prototyping Tools

- **FAISS**: Lightweight, not a full DB (testing only)  
- **Weaviate**: Open-source vector database  
- **Pinecone**: Proprietary, more costly  
- **Milvus**: Open-source similarity search engine  
- **Qdrant**: Modern, open-source vector DB  
- **Chroma**: Great for testing; still maturing  
- **LanceDB**: _To be researched_

---

### 🧾 Selected Stack

**Strategy:** Self-hosted & Open-source

- Avoiding `pgvector` due to lack of existing Postgres infra
- **Selected:** **ChromaDB** – good for testing, lightweight, no Postgres dependency

---

### Document Handling

**PDF Processing**  
- Chunked using `RecursiveCharacterTextSplitter`  
- Embedded via OpenAI’s `text-embedding-3-small`  
- Alternative embeddings can be tested via [HuggingFace Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

**Images in PDFs**  
- Extracted separately  
- Understood using vision models  
- Indexed with multimodal metadata

**Forms**  
- _Planned but not implemented yet_

---

### 🔧 Other Considerations

- **Conversation Memory**  
  - Uses `ConversationSummaryBufferMemory`  
  - Summarizes past interactions after token limits are exceeded

- **Code Execution Support**  
  - Python code questions are handled using the REPL Python tool

- **User Uploads**  
  - Users can upload documents via the Gradio interface  
  - Once indexed, documents are immediately available for querying

### Launching the service

Navigate to [app.py](./1.chatbot/src/app.py) and with the environment setup, proceed with:
````
python app.py
````
Then in the browser go to [http://127.0.0.1:7860](http://127.0.0.1:7860)

You can create a `conversation` and continue from there or directly jump in chat mode, a conversation will be created automatically. 


## 2. [Q&A](./2.QA/)

Refer to the [Q&A document](./2.QA/qa.md).


## 3. [Object Detection (Vision)](./3.Vision/)

The app is a `FastAPI` application containerized in a Docker image. The `detection` endpoint exposes a YOLO v9-e focused on two classes: human and car.


Start by building the docker.
````
docker build -t yolov9-detector .
````

Then run it
````
docker run -p 8000:8000 yolov9-detector
````
After that you may send a `GET` request directly from ``postman`` or use the  [Visualize notebook](./3.Vision/notebooks/visualize.ipynb) to interact. 

The request follows the schema:

```json
{
  "method": "GET",
  "url": "http://localhost:8000/detection",
  "query_parameters": {
    "image_url": "https://yourimage.com/image.png"
  }
}
```

The expected output with Code `200` is :

````json
{
  "message": "Detection completed",
  "detections": {
    "objects": [
      {
        "class_": "string",
        "bbox": {
          "x_center": "float",
          "y_center": "float",
          "width": "float",
          "height": "float"
        }
      }
      // ... more detected objects
    ]
  }
}
````
