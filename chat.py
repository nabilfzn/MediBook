import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File
from pydantic import BaseModel
import uuid
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import shutil

load_dotenv()
gemini_api = os.getenv("GEMINI_API_KEY")
pinecone = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# membangun chatbot
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=2000,
    timeout=None,
    max_retries=2,
    google_api_key=gemini_api
)

# Search the dense index
dense_index = pc.Index(os.getenv("INDEX_NAME"))

chat_histories = {}


def ask(query: str, user_id: str, namespace="example-namespace"):

    chat_history = chat_histories.get(user_id, [])

    # ambil hasil dari Pinecone
    results = dense_index.search(
        namespace=namespace,
        query={
            "top_k": 10,
            "inputs": {"text": query}
        }
    )

    # gabungkan hasil ke dalam context
    context = [hit["fields"]["text"] for hit in results["result"]["hits"]]
    context_text = "\n".join(context)



    # siapkan prompt messages
    messages = [
        SystemMessage(content=f"jawab dengan singkat, padat, dan gunakan informasi ini:\n {context_text} \n jika informasi tidak ada berilah respon 'maaf kita tidak memiliki informasi tersebut'"),
        *chat_history,
        HumanMessage(content=query)
    ]

    # panggil LLM
    response = chat.invoke(messages)

    # update history dengan message baru
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response.content))
    chat_histories[user_id] = chat_history

    return response.content




async def get_uid(request: Request):
    if "user_id" in request.cookies:
        return request.cookies["user_id"]
    return None

class AskRequest(BaseModel):
    query: str

@app.post('/askk')
async def tanya(body: AskRequest, request: Request, response: Response):
    query = body.query
    user_id = await get_uid(request)

    # kalau belum ada user_id → buat baru dan set ke cookie
    if not user_id:
        user_id = str(uuid.uuid4())
        response.set_cookie(key="user_id", value=user_id, httponly=True)

    if not query:
        raise HTTPException(status_code=400, detail="Query tidak boleh kosong")
    
    # misal fungsi ask()
    jawaban = ask(query=query, user_id=user_id)

    return {
        "answer": jawaban,
        "user_id": user_id
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        temp_file_path = f"data/{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load and process the PDF
        loader = PyPDFLoader(temp_file_path)
        document = loader.load()
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(document)
        
        
        # Prepare records for Pinecone
        records = []
        for i, doc in enumerate(texts):
            records.append({
                "_id": f"rec{i}",
                "text": doc.page_content
            })
        
        # Upload to Pinecone
        dense_index.upsert_records("example-namespace", records)
        
        # Clean up the temporary file
        os.remove(temp_file_path)
        
        return {
            "message": f"Successfully processed PDF: {file.filename}",
            "chunks_created": len(texts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



