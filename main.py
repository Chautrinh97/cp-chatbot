from datetime import datetime, timezone
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    BackgroundTasks
)
from http import HTTPStatus
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
import asyncio
import nest_asyncio
import shutil

from llama_index.core import VectorStoreIndex, Settings, StorageContext, Document
from llama_index.core.retrievers import VectorIndexAutoRetriever, VectorIndexRetriever, QueryFusionRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface_optimum import OptimumEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.lmstudio import LMStudio
from llama_index.llms.ollama import Ollama
from llama_index.core import StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from llama_index.core.chat_engine import ContextChatEngine
from qdrant_client import QdrantClient, AsyncQdrantClient

from utils import get_doc_from_digital_ocean, get_prompt, set_metadata_from_request, get_vector_store_info
from request_dto import SyncRequest, UnsyncRequest, RemoveAndSyncRequest, ResyncRequest
from database import get_db

load_dotenv()
nest_asyncio.apply()




#####################                   Global variables                 #####################
#####################                   Global variables                 #####################
#####################                   Global variables                 #####################




COUNT = 0
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
parser_3 = LlamaParse(
    result_type="markdown",
    auto_mode_trigger_on_table_in_page=True,
    language="vi",
    split_by_page=False,
    api_key=os.getenv("LLAMAPARSE_API_KEY_3"),
)
node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=192)
qdrant_collection_name = "qdrant_vector_store_collection"




############################                SETUP                   ###########################
############################                SETUP                   ###########################
############################                SETUP                   ###########################





def generateDataSource():
    print("2. IN LOADING INDEX PROGRESS")
    global INDEX

    # create qdrant client with endpoint and api key
    client_qdrant = QdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"), 
        api_key=os.getenv("QDRANT_APIKEY")
    )
    
    aclient_qdrant = AsyncQdrantClient(
        url=os.getenv("QDRANT_ENDPOINT"), 
        api_key=os.getenv("QDRANT_APIKEY")
    )

    # check if existed collection in qdrant client cluster
    if client_qdrant.collection_exists(collection_name=qdrant_collection_name):
        vector_store_qdrant = QdrantVectorStore(
            client=client_qdrant,
            aclient=aclient_qdrant,
            collection_name=qdrant_collection_name,
        )
        INDEX = VectorStoreIndex.from_vector_store(
            vector_store=vector_store_qdrant,
            embed_model=EMBEDDING_MODEL,
        )

    # if no, create a default document and add to INDEX, save to qdrant collection
    else:
        vector_store_qdrant = QdrantVectorStore(
            client=client_qdrant,
            aclient=aclient_qdrant,
            collection_name=qdrant_collection_name,
        )
        storage_context_qdrant = StorageContext.from_defaults(
            vector_store=vector_store_qdrant
        )
        default_docs = [
            Document(text="#This is default docs, there is nothing in here.")
        ]
        default_nodes = node_parser.get_nodes_from_documents(default_docs)
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
        temperature=0.7,
        max_tokens=2048,
    )

    # Theo kieu optium (model da duoc tai ve)
    EMBEDDING_MODEL = OptimumEmbedding(folder_name="./sentence-transformers")
    Settings.embed_model = EMBEDDING_MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    pool = await get_db()
    app.state.pool = pool
    print("1. IN STARTING UP")

    try:
        load_llm_embedding_model()
        generateDataSource()
        print("3. DONE LOADING RESOURCE")
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





######################                    MUTATION INDEX                  #####################
######################                    MUTATION INDEX                  #####################
######################                    MUTATION INDEX                  #####################






@app.post("/document/sync")
async def sync(request: SyncRequest):
    global INDEX, COUNT
    document: Document = None
    print("---ACCESS---SYNC---DOCUMENT")
    
    try:
        COUNT+=1
        if COUNT%3 == 0:
            document = await get_doc_from_digital_ocean(
                parser=parser_1, fileUrl=request.fileUrl
            )
        elif COUNT%3 == 1:
            document = await get_doc_from_digital_ocean(
                parser=parser_2, fileUrl=request.fileUrl
            )
        else:
            document = await get_doc_from_digital_ocean(
                parser=parser_3, fileUrl=request.fileUrl
            )
            
    except:
        print("---SYNC---PARSE-DOCUMENT---FAILED---")
        raise HTTPException(HTTPStatus.BAD_REQUEST)
    
    set_metadata_from_request(request, document)
    for key, value in document.metadata.items():
        print(f"Key: {key} --- Value: {value}")

    doc_id = document.doc_id
    nodes = node_parser.get_nodes_from_documents([document])
    
    try:
        INDEX.insert_nodes(nodes)
    except:
        print("---SYNC---ADD-TO-QDRANT---FAILED---")
        raise HTTPException(HTTPStatus.BAD_REQUEST)
    
    os.makedirs(request.key)
    file_path = os.path.join(request.key, 'file.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(document.text)
    print("---SYNC---SUCCESSFULLY---")
    return {"doc_id": doc_id, "message": "Successfully sync document"}


@app.post("/document/remove-and-resync")
async def remove_and_resync(request: RemoveAndSyncRequest):
    global INDEX, COUNT
    document: Document = None
    print("---ACCESS---REMOVE-AND-SYNC---DOCUMENT")
    try:
        COUNT+=1
        if COUNT%3 == 0:
            document = await get_doc_from_digital_ocean(
                parser=parser_1, fileUrl=request.fileUrl
            )
        elif COUNT%3 == 1:
            document = await get_doc_from_digital_ocean(
                parser=parser_2, fileUrl=request.fileUrl
            )
        else:
            document = await get_doc_from_digital_ocean(
                parser=parser_3, fileUrl=request.fileUrl
            )

    except:
        print("---REMOVE-AND-SYNC---PARSE-DOCUMENT---FAILED---")
        raise HTTPException(HTTPStatus.BAD_REQUEST)
    
    set_metadata_from_request(request, document)
    for key, value in document.metadata.items():
        print(f"Key: {key} --- Value: {value}")
            
    doc_id = document.doc_id
    nodes = node_parser.get_nodes_from_documents([document])
        
    try:
        INDEX.insert_nodes(nodes)
    except:
        print("---REMOVE-AND-SYNC---ADD-TO-QDRANT---FAILED---")
        raise HTTPException(HTTPStatus.BAD_REQUEST)
        
    shutil.rmtree(request.old_key) 
    
    INDEX.delete_ref_doc(request.doc_id)  
             
    os.makedirs(request.key)
    file_path = os.path.join(request.key, 'file.txt')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(document.text)

    print("---REMOVE-AND-SYNC---SUCCESSFULLY")
    return {"doc_id": doc_id, "message": "Successfully remove and sync document"}


@app.post("/document/unsync")
async def unsync(request: UnsyncRequest):
    global INDEX
    print("---ACCESS---UNSYNC---DOCUMENT")
        
    shutil.rmtree(request.key)
    
    INDEX.delete_ref_doc(ref_doc_id=request.doc_id)
    return {"message": "Successfully unsync document"}


@app.post("/document/resync")
async def resync(request: ResyncRequest):
    document = SimpleDirectoryReader(input_dir=request.key).load_data()[0]
    set_metadata_from_request(request, document)
    doc_id = document.doc_id
    nodes = node_parser.get_nodes_from_documents([document])
        
    try:
        INDEX.insert_nodes(nodes)
    except:
        print("---RESYNC---DOCUMENT---FAILED---")
        raise HTTPException(HTTPStatus.BAD_REQUEST)
    
    INDEX.delete_ref_doc(request.doc_id)
    print("---RESYNC---DOCUMENT---SUCCESSFULLY---")
    return {"doc_id": doc_id, "message": "Successfully remove and sync document"}
    
    

    
    
######################                    CHAT AREA                   #####################
######################                    CHAT AREA                   #####################
######################                    CHAT AREA                   #####################





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
    
    #Auto retriver (with metadata)
    retriever_auto = VectorIndexAutoRetriever(
        index=INDEX,
        vector_store_info=get_vector_store_info(),
        llm=LLM['Gemini'],
        similarity_top_k=10,
    )
    
    # Default retriever
    default_retriever = VectorIndexRetriever(
        index=INDEX,
        similarity_top_k=10,
    )
    
    retriever = QueryFusionRetriever(
        retrievers=[default_retriever, retriever_auto],
        num_queries=2,
        llm=LLM['Gemini'],
        mode='relative_score',
    )
    
    engine = ContextChatEngine.from_defaults(
        retriever=retriever,
        llm=LLM["Gemini"],
        system_prompt=get_prompt(),
    )
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
        
        await save_message_to_db(question, "user", conversation_id, pool)

        chunks = ""

        async def run_engine():
            nonlocal chunks
            engine = await create_engine()
            response = engine.stream_chat(question)
            for token in response.response_gen:
                await websocket.send_text(token)
                chunks += token
                await asyncio.sleep(0.2)
            print("Complete streaming")

        # Start streaming data to client
        await run_engine()
        await save_message_to_db(chunks, "assistant", conversation_id, pool)
        await websocket.close()

    except WebSocketDisconnect:
        print("Client disconnected -> end streaming")
        await save_message_to_db(chunks, "assistant", conversation_id, pool)
    except Exception as e:
        await websocket.close()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@app.websocket("/non-save-chat")
async def non_save_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Lắng nghe message từ client
        data = await websocket.receive_json()
        question = data.get("question")

        async def run_engine():
            engine = await create_engine()
            response = engine.stream_chat(question)
            for node in response.source_nodes:
                print("Node score:", node.text)
            for token in response.response_gen:
                print(token)
                await websocket.send_text(token)
                await asyncio.sleep(0.2)
            print("Complete streaming")

        # Start streaming data to client
        await run_engine()
        await websocket.close()

    except WebSocketDisconnect:
        print("Client disconnected -> end streaming")
    except Exception as e:
        await websocket.close()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
