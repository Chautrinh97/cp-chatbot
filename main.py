from datetime import datetime, timezone
from typing import AsyncGenerator
from fastapi import (
    FastAPI, 
    HTTPException, 
    WebSocket,
    WebSocketDisconnect,
    HTTPException
)
from http import HTTPStatus
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
import qdrant_client
from llama_index.core.chat_engine import ContextChatEngine

from utils import getDocumentFromDigitalOcean, get_prompt, set_metadata_from_request
from request_dto import (
    SyncRequest,
    UnsyncRequest,
    RemoveAndSyncRequest,
)
from database import get_db

load_dotenv()
nest_asyncio.apply()

##################### Global variables #####################
##################### Global variables #####################
##################### Global variables #####################

EMBEDDING_MODEL = None
LLM = {}
INDEX: VectorStoreIndex = None
parser_1 = LlamaParse(
    result_type="markdown",
    auto_mode_trigger_on_table_in_page=True,
    language="vi",
    split_by_page=False,
    api_key=os.getenv("LLAMAPARSE_API_KEY_1"),
)
parser_2 = LlamaParse(
    result_type="markdown",
    auto_mode_trigger_on_table_in_page=True,
    language="vi",
    split_by_page=False,
    api_key=os.getenv("LLAMAPARSE_API_KEY_2"),
)
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=128)

############################                SETUP                   ###########################
############################                SETUP                   ###########################
############################                SETUP                   ###########################

def generateDataSource():
    print("2. HAVING ACESSED THIS LOAD INDEX PROGRESS")
    global INDEX, BM25

    # create qdrant client with endpoint and api key
    client_qdrant = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"), api_key=os.getenv("QDRANT_APIKEY")
    )

    # create a vector_store in qdrant with collection_name
    collection_name = "qdrant_vector_store_collection"

    # check if existed collection in qdrant client cluster
    if client_qdrant.collection_exists(collection_name=collection_name):
        vector_store_qdrant = QdrantVectorStore(
            client=client_qdrant, collection_name=collection_name
        )
        storage_context_qdrant = StorageContext.from_defaults(
            vector_store=vector_store_qdrant
        )
        INDEX = VectorStoreIndex.from_vector_store(
            vector_store=vector_store_qdrant, embed_model=EMBEDDING_MODEL
        )

    # if no, create a default document and add to INDEX, save to qdrant collection
    else:
        default_docs = [
            Document(text="#This is default docs, there is nothing in here.")
        ]
        default_nodes = node_parser.get_nodes_from_documents(default_docs)
        vector_store_qdrant = QdrantVectorStore(
            client=client_qdrant, collection_name=collection_name
        )
        storage_context_qdrant = StorageContext.from_defaults(
            vector_store=vector_store_qdrant
        )
        INDEX = VectorStoreIndex(
            nodes=default_nodes,
            storage_context=storage_context_qdrant,
            embed_model=EMBEDDING_MODEL,
        )

def load_llm_embedding_model():
    global LLM, EMBEDDING_MODEL
    
    LLM["Gemini"] = Gemini(
        model=os.getenv("GEMINI_MODEL"),
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.5,
        max_tokens=1024,
    )
       
    #Theo kieu inference
    """ embedding_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ) """
    
    #Theo kieu optium (model da duoc tai ve)
    EMBEDDING_MODEL = OptimumEmbedding(folder_name="./sentence-transformers")
    Settings.embed_model = EMBEDDING_MODEL

@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_db()
    app.state.pool = pool
    print("1. HAVING ACCESSED START UP")

    try:
        load_llm_embedding_model()
        generateDataSource()

        print("3. HAVE DONE LOADING RESOURCE") 
        yield

    except Exception as e:
        print(f"Error during lifespan initialization: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to initialize application resources."
        )
    finally:
        pool.close()
        print("Shutting down application...")


app = FastAPI(lifespan=lifespan)

######################                    FUNCTION                  #####################
######################                    FUNCTION                  #####################
######################                    FUNCTION                  #####################

@app.post("/document/sync")
async def sync(request: SyncRequest):
    global INDEX
    document = None
    print("---ACCESS---SYNC---DOCUMENT")
    try:
        document = await getDocumentFromDigitalOcean(
            parser=parser_1, fileUrl=request.fileUrl
        )
        set_metadata_from_request(request, document)
        for key, value in document.metadata.items():
            print(f"Key: {key} --- Value: {value}")
    except:
        print("---SYNC---PARSE-DOCUMENT---FAILED---")
        raise HTTPException(HTTPStatus.BAD_REQUEST)

    doc_id = document.doc_id
    nodes = node_parser.get_nodes_from_documents([document])
    INDEX.insert_nodes(nodes)

    print("---SYNC---SUCCESSFULLY---")
    return {"doc_id": doc_id, "message": "Successfully sync document"}


@app.post("/document/remove-and-sync")
async def remove_and_sync(request: RemoveAndSyncRequest):
    global INDEX
    document = None
    print("---ACCESS---REMOVE-AND-SYNC---DOCUMENT")
    try:
        document = await getDocumentFromDigitalOcean(
            parser=parser_2, fileUrl=request.fileUrl
        )
        set_metadata_from_request(request, document)
        for key, value in document.metadata.items():
            print(f"Key: {key} --- Value: {value}")
    except:
        print("---REMOVE-AND-SYNC---PARSE-DOCUMENT---FAILED---")
        raise HTTPException(HTTPStatus.BAD_REQUEST)

    doc_id = document.doc_id
    nodes = node_parser.get_nodes_from_documents([document])
    INDEX.delete_ref_doc(request.doc_id)
    INDEX.insert_nodes(nodes)

    print("---REMOVE-AND-SYNC---SUCCESSFULLY")
    return {"doc_id": doc_id, "message": "Successfully remove and sync document"}


@app.post("/document/unsync")
async def unsync(request: UnsyncRequest):
    global INDEX
    print("---ACCESS---UNSYNC---DOCUMENT")
    INDEX.delete_ref_doc(ref_doc_id=request.doc_id)

    return {"message": "Successfully unsync document"}


""" @app.post("/chat")
async def chat_endpoint(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    conversation_id = data.get("conversation_id")
    question = data.get("question")
    pool = await get_db()
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                # Insert question into Message table
                await cursor.execute(
                    "INSERT INTO messages (conversation_id, content, role) VALUES (%s, %s, %s)",
                    (conversation_id, question, "user"),
                )
                # Update conversation's updatedAt timestamp
                updated_at = datetime.now(timezone.utc)
                await cursor.execute(
                    "UPDATE conversations SET updated_at = %s WHERE id = %s",
                    (updated_at, conversation_id),
                )
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


    async def streaming_simulation():
        chunks = ""
        for i in range(1, 20):
            chunks += str(i) + " "
            yield str(i) + " "
            await asyncio.sleep(0.5)
    
        await save_answer_to_db(chunks, conversation_id)

    return StreamingResponse(streaming_simulation(), media_type="text/event-stream") """


async def save_message_to_db(message: str, role: str, conversation_id, pool):
    if not message.strip():
        return
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            try:
                # Insert response into Message table
                await cursor.execute(
                    "INSERT INTO messages (conversation_id, content, role) VALUES (%s, %s, %s)",
                    (conversation_id, message, role),
                )
                # Update conversation's updated_at timestamp
                updated_at = datetime.now(timezone.utc)
                await cursor.execute(
                    "UPDATE conversations SET updated_at = %s WHERE id = %s",
                    (updated_at, conversation_id),
                )
                await conn.commit()
            except Exception as e:
                await conn.rollback()
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
                        
async def create_engine():
    
    engine = ContextChatEngine.from_defaults(
        retriever=INDEX.as_retriever(similarity_top_k=5),
        llm=LLM["Gemini"],
        system_prompt=get_prompt(),
    )
    
    """ engine = INDEX.as_chat_engine(
        chat_mode="context",
        llm=LLM["Gemini"],
        context_prompt=get_prompt(),
        similarity_top_k=5,
        verbose=True,
    ) """
    return engine
            
@app.websocket("/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Lắng nghe message từ client
        data = await websocket.receive_json()
        conversation_id = data.get("conversation_id")
        question = data.get("question")

        pool = app.state.pool
        await save_message_to_db(question, 'user', conversation_id, pool)

        chunks = ""
        async def run_engine():
            nonlocal chunks
            engine = await create_engine()
            response = engine.stream_chat(question)
            for node in response.source_nodes:
                print("Node score:", node.score)
            for token in response.response_gen:
                await websocket.send_text(token)
                chunks+=token
                await asyncio.sleep(0.1)
            print("Complete streaming")

        # Start streaming data to client
        await run_engine()
        await save_message_to_db(chunks, 'assistant', conversation_id, pool)
        await websocket.close()
        
    except WebSocketDisconnect:
        print("Client disconnected -> end streaming")
        await save_message_to_db(chunks, 'assistant', conversation_id, pool)
    except Exception as e:
        await websocket.close()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
