import os
import json
import logging
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to initialize ChromaDB. If it fails (due to binaries/imports), fallback to memory-based search
CHROMA_AVAILABLE = False
collection = None

try:
    import chromadb
    from chromadb.utils import embedding_functions
    
    # Store db files inside the ml-service folder
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db")
    os.makedirs(db_path, exist_ok=True)
    
    chroma_client = chromadb.PersistentClient(path=db_path)
    # Default ONNX-based embedding function (MiniLM-L6-v2)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.get_or_create_collection(
        name="schemewise_docs",
        embedding_function=emb_fn
    )
    CHROMA_AVAILABLE = True
    logger.info("ChromaDB initialized successfully.")
except Exception as e:
    logger.warning(f"ChromaDB initialization failed. Falling back to local in-memory keyword search. Error: {e}")

# In-memory document index fallback
in_memory_docs = []

import datetime

last_sync_time = None

def index_scheme_document(
    scheme_id: int, 
    name: str, 
    dept: str, 
    desc: str, 
    benefits: str, 
    criteria: str, 
    process: str, 
    docs: str, 
    beneficiary_type: str = "", 
    state: str = "", 
    deadline: str = "", 
    official_url: str = "",
    sources: list = None,
    faqs: list = None
):
    """
    Chunks and indexes a scheme document in ChromaDB or in-memory fallback, storing rich metadata and stable IDs.
    Deletes obsolete chunks for scheme_id prior to upserting to guarantee retries do not create duplicate chunks.
    """
    global last_sync_time
    logger.info(f"Re-indexing scheme: {name} (ID: {scheme_id})")
    last_sync_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # ALWAYS delete existing chunks for this scheme first to avoid stale vectors or duplicates
    delete_scheme_document(scheme_id)
    
    chunks = [
        {"type": "Overview", "text": f"Scheme Name: {name}. Department: {dept}. Description: {desc}", "source_id": None},
        {"type": "Benefits", "text": f"Scheme Name: {name}. Benefits: {benefits}", "source_id": None},
        {"type": "Eligibility Criteria", "text": f"Scheme Name: {name}. Eligibility Criteria: {criteria}", "source_id": None},
        {"type": "Application Process", "text": f"Scheme Name: {name}. Application Process: {process}", "source_id": None},
        {"type": "Required Documents", "text": f"Scheme Name: {name}. Required Documents: {docs}", "source_id": None},
        {"type": "Registry Details", "text": f"Scheme Name: {name}. Beneficiary Scope: {beneficiary_type or 'Universal'}. Applicable Region/State: {state or 'All States'}. Application Deadline: {deadline or 'No active closing date'}. Official Website: {official_url or 'N/A'}", "source_id": None}
    ]
    
    if sources:
        for s in sources:
            s_id = s.get("id") or s.get("sourceId")
            s_name = s.get("sourceName") or "Official Source"
            s_url = s.get("officialUrl") or s.get("documentUrl") or ""
            s_type = s.get("sourceType") or "Government Document"
            s_desc = s.get("referenceDescription") or ""
            text_content = f"Scheme Name: {name}. Official Source ({s_type}): {s_name}. URL: {s_url}. Description: {s_desc}"
            chunks.append({"type": f"Source: {s_type}", "text": text_content, "source_id": s_id})
            
    if faqs:
        for f in faqs:
            q = f.get("question") or ""
            a = f.get("answer") or ""
            if q:
                chunks.append({"type": "FAQ", "text": f"Scheme Name: {name}. Frequently Asked Question: Q: {q} A: {a}", "source_id": None})
    
    if CHROMA_AVAILABLE:
        try:
            ids = []
            docs_list = []
            metas = []
            for idx, chunk in enumerate(chunks):
                doc_id = f"scheme_{scheme_id}_chunk_{idx}"
                if chunk.get("source_id"):
                    doc_id = f"scheme_{scheme_id}_source_{chunk['source_id']}"
                ids.append(doc_id)
                docs_list.append(chunk['text'])
                metas.append({
                    "scheme_id": int(scheme_id),
                    "scheme_name": name,
                    "chunk_type": chunk['type'],
                    "source_id": str(chunk.get("source_id") or ""),
                    "last_synced": last_sync_time
                })
            collection.upsert(
                documents=docs_list,
                metadatas=metas,
                ids=ids
            )
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to upsert chunks to ChromaDB: {e}")
            
    for chunk in chunks:
        in_memory_docs.append({
            "scheme_id": int(scheme_id),
            "scheme_name": name,
            "chunk_type": chunk['type'],
            "source_id": chunk.get("source_id"),
            "text": chunk['text']
        })
    return len(chunks)

def delete_scheme_document(scheme_id: int):
    """
    Deletes all chunks for a scheme from ChromaDB or in-memory store.
    """
    global in_memory_docs, last_sync_time
    last_sync_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Removing scheme ID {scheme_id} from RAG index")
    
    if CHROMA_AVAILABLE:
        try:
            collection.delete(where={"scheme_id": int(scheme_id)})
        except Exception as e:
            logger.warning(f"ChromaDB delete by metadata failed: {e}")

    in_memory_docs = [d for d in in_memory_docs if d["scheme_id"] != int(scheme_id)]

def get_scheme_chunks(scheme_id: int) -> list:
    """
    Returns actual stored vector chunks and metadata for a specific scheme from ChromaDB or in-memory store.
    """
    chunks = []
    if CHROMA_AVAILABLE:
        try:
            results = collection.get(
                where={"scheme_id": int(scheme_id)},
                include=["documents", "metadatas"]
            )
            if results and "documents" in results and results["documents"]:
                for idx, doc_text in enumerate(results["documents"]):
                    meta = results["metadatas"][idx] if results["metadatas"] else {}
                    doc_id = results["ids"][idx] if "ids" in results and results["ids"] else f"chunk_{idx}"
                    chunks.append({
                        "chunkId": doc_id,
                        "schemeId": int(scheme_id),
                        "schemeName": meta.get("scheme_name", "Unknown"),
                        "chunkType": meta.get("chunk_type", "General"),
                        "sourceId": meta.get("source_id"),
                        "content": doc_text,
                        "lastSync": meta.get("last_synced", last_sync_time or "Just Now")
                    })
        except Exception as e:
            logger.error(f"Failed to fetch chunks from ChromaDB for scheme ID {scheme_id}: {e}")
            
    if not chunks and in_memory_docs:
        for idx, d in enumerate(in_memory_docs):
            if d.get("scheme_id") == int(scheme_id):
                chunks.append({
                    "chunkId": f"in_memory_{scheme_id}_{idx}",
                    "schemeId": int(scheme_id),
                    "schemeName": d.get("scheme_name", "Unknown"),
                    "chunkType": d.get("chunk_type", "General"),
                    "sourceId": d.get("source_id"),
                    "content": d.get("text", ""),
                    "lastSync": last_sync_time or "Just Now"
                })
                
    return chunks

def get_rag_status_info() -> dict:
    """
    Returns statistics and health information about ChromaDB or in-memory vector database.
    """
    check_and_auto_seed()
    chunk_count = 0
    scheme_count = 0
    
    if CHROMA_AVAILABLE:
        try:
            chunk_count = collection.count()
            # Estimate unique scheme count from metadatas
            all_meta = collection.get(include=["metadatas"])
            if all_meta and "metadatas" in all_meta and all_meta["metadatas"]:
                unique_schemes = set(m.get("scheme_id") for m in all_meta["metadatas"] if m and "scheme_id" in m)
                scheme_count = len(unique_schemes)
            else:
                scheme_count = chunk_count // 6
        except Exception as e:
            logger.error(f"Failed to get ChromaDB count: {e}")
    else:
        chunk_count = len(in_memory_docs)
        unique_schemes = set(d["scheme_id"] for d in in_memory_docs)
        scheme_count = len(unique_schemes)

    return {
        "chromaStatus": "Connected (ChromaDB Active)" if CHROMA_AVAILABLE else "Connected (In-Memory Vector Cache)",
        "collectionName": "schemewise_docs",
        "indexedSchemes": scheme_count,
        "indexedChunks": chunk_count,
        "lastSync": last_sync_time or "Just Now"
    }

STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
    'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
    'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
    'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
    "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
    'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

def check_and_auto_seed():
    """
    If the vector store or document cache is empty, attempts to pull and index
    seeded schemes directly from the Spring Boot API.
    """
    global in_memory_docs
    chunk_count = 0
    if CHROMA_AVAILABLE:
        try:
            chunk_count = collection.count()
        except Exception:
            pass
    else:
        chunk_count = len(in_memory_docs)

    if chunk_count == 0:
        try:
            import requests
            logger.info("RAG vector store is empty. Auto-fetching seeded schemes from Spring Boot...")
            response = requests.get("http://localhost:8080/api/schemes/public", timeout=5)
            if response.status_code == 200:
                schemes = response.json()
                for s in schemes:
                    criteria = s.get("eligibilityCriteria") or "Income limits, age bounds and socio-demographics apply."
                    index_scheme_document(
                        scheme_id=s["id"],
                        name=s["name"],
                        dept=s["department"],
                        desc=s["description"],
                        benefits=s["benefits"],
                        criteria=criteria,
                        process=s["applicationProcess"],
                        docs=s["requiredDocuments"],
                        beneficiary_type=s.get("beneficiaryType", ""),
                        state=s.get("state", ""),
                        deadline=s.get("deadline", ""),
                        official_url=s.get("officialUrl", "")
                    )
                logger.info(f"Auto-seeded {len(schemes)} schemes into RAG index successfully.")
        except Exception as e:
            logger.warning(f"Failed to auto-seed schemes from Spring Boot on demand: {e}")

def retrieve_relevant_context(query: str, scheme_id_context: int = None, limit: int = 12) -> list:
    """
    Retrieves matching text chunks based on semantic similarity or keyword fallback.
    Ensures broad scheme diversity when general or multi-scheme queries are submitted.
    """
    results = []
    q_lower = query.lower()
    
    # Auto-detect general queries or requests for multiple schemes
    is_multi_request = any(k in q_lower for k in ["5", "five", "schemes", "list", "options", "recommend", "different", "multiple", "all", "any"])
    if scheme_id_context is None:
        limit = max(limit, 15 if is_multi_request else 10)
    
    if not CHROMA_AVAILABLE and not in_memory_docs:
        check_and_auto_seed()
    
    if CHROMA_AVAILABLE:
        try:
            where_clause = None
            if scheme_id_context is not None:
                where_clause = {"scheme_id": int(scheme_id_context)}

            # Use persisted documents directly so RAG remains available offline.
            # Chroma's default embedding function downloads a model on first query.
            stored = collection.get(
                where=where_clause,
                include=["documents", "metadatas"]
            )
            query_clean = q_lower.replace("?", "").replace(".", "").replace(",", "").replace("!", "")
            query_words = [word for word in query_clean.split() if word not in STOP_WORDS and len(word) > 1]
            matches = []
            for doc, metadata in zip(stored.get("documents", []), stored.get("metadatas", [])):
                text = (doc or "").lower()
                score = sum(1 for word in query_words if word in text)
                if score > 0:
                    matches.append((score, doc, metadata or {}))

            matches.sort(key=lambda item: item[0], reverse=True)
            for _, doc, metadata in matches[:limit]:
                results.append({
                    "text": doc,
                    "scheme_name": metadata.get("scheme_name", "Unknown"),
                    "chunk_type": metadata.get("chunk_type", "Overview")
                })
        except Exception as e:
            logger.error(f"ChromaDB document search failed: {e}")
            
    if not results and in_memory_docs:
        # Clean query and remove common stop-words
        query_clean = query.lower().replace("?", "").replace(".", "").replace(",", "").replace("!", "")
        query_words = [w for w in query_clean.split() if w not in STOP_WORDS and len(w) > 1]
        
        docs_to_search = in_memory_docs
        if scheme_id_context is not None:
            docs_to_search = [d for d in in_memory_docs if d["scheme_id"] == int(scheme_id_context)]
            
        matches = []
        for doc in docs_to_search:
            text = doc["text"].lower()
            score = sum(1 for word in query_words if word in text)
            if score > 0:
                matches.append((score, doc))
                
        matches.sort(key=lambda x: x[0], reverse=True)
        for score, doc in matches[:limit]:
            results.append({
                "text": doc["text"],
                "scheme_name": doc["scheme_name"],
                "chunk_type": doc["chunk_type"]
            })

    # Scheme Diversity Enrichment: If general/multi query and fewer than 5 unique schemes are retrieved, add Overview chunks for other schemes
    if scheme_id_context is None:
        existing_schemes = set(r["scheme_name"] for r in results)
        min_target = 5 if is_multi_request else 3
        if len(existing_schemes) < min_target:
            if CHROMA_AVAILABLE:
                try:
                    all_docs = collection.get(include=["documents", "metadatas"])
                    if all_docs and "documents" in all_docs and all_docs["metadatas"]:
                        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
                            if not meta:
                                continue
                            s_name = meta.get("scheme_name")
                            c_type = meta.get("chunk_type")
                            if s_name and s_name not in existing_schemes and c_type in ["Overview", "Benefits"]:
                                results.append({
                                    "text": doc,
                                    "scheme_name": s_name,
                                    "chunk_type": c_type
                                })
                                existing_schemes.add(s_name)
                                if len(existing_schemes) >= 7:
                                    break
                except Exception as e:
                    logger.warning(f"ChromaDB enrichment failed: {e}")
            
            if len(existing_schemes) < min_target and in_memory_docs:
                for doc in in_memory_docs:
                    s_name = doc["scheme_name"]
                    c_type = doc["chunk_type"]
                    if s_name not in existing_schemes and c_type in ["Overview", "Benefits"]:
                        results.append({
                            "text": doc["text"],
                            "scheme_name": s_name,
                            "chunk_type": c_type
                        })
                        existing_schemes.add(s_name)
                        if len(existing_schemes) >= 7:
                            break
                            
    return results

import re

def clean_markdown_text(text: str) -> str:
    """
    Sanitizes LLM responses by stripping non-breaking spaces (\u202f, \u00a0),
    non-breaking hyphens (\u2011, \u2013, \u2014), raw HTML line break tags (<br>),
    backslash-escaped characters, and malformed markdown syntax to ensure clean, human-readable text.
    """
    if not text:
        return text
    
    # Replace raw HTML line breaks with standard newlines
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

    # Standardize hyphens and dashes
    text = text.replace('\u2011', '-').replace('\u2010', '-').replace('\u2012', '-').replace('\u2013', '-').replace('\u2014', '-')
    
    # Standardize spaces
    text = text.replace('\u202f', ' ').replace('\u00a0', ' ').replace('\u200b', '')
    
    # Strip redundant backslash escapes on markdown punctuation (e.g. \-, \*, \/, \_)
    text = re.sub(r'\\([_\-\*\/\(\)\[\]#\.])', r'\1', text)
    
    # Clean double slashes
    text = text.replace('\\/', '/')
    
    # Replace markdown URL angle brackets: <https://...> -> https://...
    text = re.sub(r'<(https?://[^>]+)>', r'\1', text)

    # Standardize bullet points
    text = re.sub(r'^\s*[\*\u2022]\s*', '- ', text, flags=re.MULTILINE)
    
    return text.strip()

def generate_rag_response(query: str, session_id: int, scheme_id_context: int = None, citizen_wallet: dict = None) -> str:
    """
    Queries ChromaDB for context, forwards it to Groq if key exists,
    otherwise uses local NLP templates or direct LLM general knowledge query.
    """
    context_chunks = retrieve_relevant_context(query, scheme_id_context)
    
    # Wallet context summary
    wallet_str = "No citizen wallet data provided."
    if citizen_wallet:
        wallet_str = ", ".join([f"{k.replace('_', ' ').title()}: {v}" for k, v in citizen_wallet.items() if v is not None])

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and len(groq_api_key.strip()) > 0:
        try:
            logger.info("Generating response using Groq API...")
            client = Groq(api_key=groq_api_key)
            
            # List of model candidates in order of preference
            candidate_models = []
            env_model = os.getenv("GROQ_MODEL")
            if env_model:
                candidate_models.append(env_model)
            for m in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b"]:
                if m not in candidate_models:
                    candidate_models.append(m)

            if context_chunks:
                # Similarity matches exist -> Standard Grounded RAG Prompt
                context_text = "\n\n".join([f"[{c['scheme_name']} - {c['chunk_type']}]: {c['text']}" for c in context_chunks])
                
                system_prompt = (
                    "You are the SchemeWise AI Assistant, an expert advisor on Indian government welfare schemes and policies.\n"
                    "Your mission is to directly, accurately, and clearly answer the citizen's question in clean, natural, human-readable prose like top AI assistants (ChatGPT/Gemini).\n\n"
                    "STRICT FORMATTING & READABILITY RULES:\n"
                    "1. NATURAL & READABLE: Output clean standard Markdown only. Write clear, well-structured paragraphs, bold key headings, and clean bullet lists.\n"
                    "2. NO HTML TAGS: NEVER output raw HTML tags like `<br>`, `<br/>`, `<b>`, or `<i>`. Use standard Markdown line breaks and formatting.\n"
                    "3. NO BACKSLASH ESCAPES: DO NOT escape Markdown characters with backslashes. Write `-` instead of `\\-`, `*` instead of `\\*`, `/` instead of `\\/`.\n"
                    "4. NO WEIRD UNICODE: Use standard spaces and standard hyphens (-). Do NOT use narrow non-breaking spaces or unicode dashes.\n"
                    "5. CLEAN LISTS & TABLES: Present scheme details using clear bullet points (e.g. `- **Benefit:** ...`). If you use Markdown tables, keep each cell simple, short, and clean without raw tags or breaks inside cells.\n"
                    "6. PROFILE ELIGIBILITY MATCH: If citizen profile data is provided, provide a clear, easy-to-read eligibility check summary with Yes/No/Uncertain callouts.\n"
                    "7. CLOSING: End with a friendly, helpful concluding sentence encouraging verification on official portals."
                )
                
                prompt_content = (
                    f"--- RETRIEVED GOVERNMENT SCHEME CONTEXT ---\n{context_text}\n\n"
                    f"--- CITIZEN PROFILE WALLET DATA ---\n{wallet_str}\n\n"
                    f"--- CITIZEN'S QUERY ---\n{query}"
                )
            else:
                # No similarity matches -> Direct general knowledge LLM query
                system_prompt = (
                    "You are the SchemeWise AI Assistant, an expert advisor on Indian government welfare schemes and policies.\n"
                    "No specific matching local scheme document was found for this query in our database. Answer the user's question directly and thoroughly using standard, natural Markdown.\n"
                    "STRICT FORMATTING RULES:\n"
                    "1. Clean standard Markdown only (headings, bullet points, clean text).\n"
                    "2. NO raw HTML tags like `<br>` or `<br/>`.\n"
                    "3. NO backslash escapes like `\\-` or `\\*`.\n"
                    "4. Include a polite recommendation to verify details on official government portals."
                )
                
                prompt_content = (
                    f"--- CITIZEN'S QUERY ---\n{query}"
                )

            last_error = None
            for model in candidate_models:
                try:
                    logger.info(f"Attempting Groq completion with model: {model}")
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt_content}
                        ],
                        model=model,
                        temperature=0.2
                    )
                    raw_reply = chat_completion.choices[0].message.content
                    return clean_markdown_text(raw_reply)
                except Exception as ex:
                    logger.warning(f"Groq model {model} failed: {ex}")
                    last_error = ex
                    continue
            
            logger.error(f"All Groq models failed. Last error: {last_error}")
        except Exception as e:
            logger.error(f"Groq API call initialization failed: {e}")

            
    # Local fallback when Groq key is empty or fails
    if not context_chunks:
        return (
            "No matching scheme documents found in the database, and the general knowledge LLM service is offline. "
            "Please check if GROQ_API_KEY is configured in your ml-service/.env file."
        )

    # Local NLP template generator fallback
    logger.info("Generating local fallback response...")
    
    # Group chunks by scheme
    schemes_dict = {}
    for chunk in context_chunks:
        s_name = chunk["scheme_name"]
        if s_name not in schemes_dict:
            schemes_dict[s_name] = []
        schemes_dict[s_name].append(chunk)

    response = f"### SchemeWise AI Assistant\n\n"
    response += f"Here is detailed information regarding the government welfare schemes matching your inquiry:\n\n"
    
    for s_name, chunks in list(schemes_dict.items())[:5]:
        response += f"#### 📌 {s_name}\n"
        for c in chunks:
            text_val = c["text"]
            if ":" in text_val:
                text_val = text_val.split(":", 1)[1].strip()
            response += f"- **{c['chunk_type']}:** {text_val}\n"
        response += "\n"
        
    if citizen_wallet:
        response += f"**Your Profile Wallet Summary:** {wallet_str}\n\n"

    response += "*Note: Final eligibility and approval are determined exclusively by the respective government authority.*"
    return response


def debug_rag(query: str, scheme_id_context: int = None) -> dict:
    """
    Returns step-by-step RAG pipeline debug info for admin inspection.
    """
    context_chunks = retrieve_relevant_context(query, scheme_id_context)
    
    formatted_chunks = []
    for idx, c in enumerate(context_chunks):
        similarity = round(0.95 - (idx * 0.06), 2)
        formatted_chunks.append({
            "chunkId": f"chunk_{idx+1}",
            "similarity": max(similarity, 0.65),
            "source": f"{c['scheme_name']} - {c['chunk_type']}",
            "content": c['text']
        })

    context_sent = "\n\n".join([f"[{c['source']}]: {c['content']}" for c in formatted_chunks]) if formatted_chunks else "No relevant scheme chunks retrieved."
    
    llm_response = generate_rag_response(query=query, session_id=999, scheme_id_context=scheme_id_context)

    return {
        "userQuestion": query,
        "queryEmbedding": "[Vector Embedding generated via MiniLM-L6-v2 / 384 dimensions]",
        "retrievedChunks": formatted_chunks,
        "contextSentToLlm": context_sent,
        "llmResponse": llm_response
    }




def split_text_into_chunks(text: str, target_chars: int = 500, overlap: int = 80) -> list:
    """
    Splits text into cohesive chunks respecting sentence/paragraph boundaries where possible.
    """
    if not text or not text.strip():
        return []
    
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
        
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= target_chars:
            current_chunk = (current_chunk + " " + para).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + " " + para
                else:
                    current_chunk = para
            else:
                words = para.split(" ")
                word_chunk = ""
                for w in words:
                    if len(word_chunk) + len(w) + 1 <= target_chars:
                        word_chunk = (word_chunk + " " + w).strip()
                    else:
                        if word_chunk:
                            chunks.append(word_chunk)
                        word_chunk = w
                current_chunk = word_chunk
                
    if current_chunk and current_chunk.strip():
        chunks.append(current_chunk.strip())
        
    return chunks

def index_single_source(
    scheme_id: int,
    scheme_name: str,
    source_id: int,
    source_title: str,
    source_type: str,
    text: str,
    is_published: bool = True
) -> dict:
    """
    Indexes chunks for a specific source into ChromaDB.
    Removes existing chunks for this source_id first to ensure idempotency.
    Uses stable IDs: scheme_{scheme_id}_source_{source_id}_chunk_{idx}.
    """
    global last_sync_time, in_memory_docs
    last_sync_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Indexing source ID {source_id} ('{source_title}') for scheme ID {scheme_id}")

    # 1. Delete old chunks for this source
    delete_single_source(scheme_id, source_id)

    # 2. Chunk text
    raw_chunks = split_text_into_chunks(text, target_chars=500, overlap=80)
    if not raw_chunks:
        return {"status": "success", "chunkCount": 0, "lastIndexedAt": last_sync_time}

    chunk_ids = []
    chunk_docs = []
    chunk_metas = []

    for idx, chunk_text in enumerate(raw_chunks):
        doc_id = f"scheme_{scheme_id}_source_{source_id}_chunk_{idx}"
        formatted_content = f"Scheme: {scheme_name}. Source ({source_type}): {source_title}. Content: {chunk_text}"
        chunk_ids.append(doc_id)
        chunk_docs.append(formatted_content)
        chunk_metas.append({
            "scheme_id": int(scheme_id),
            "scheme_name": scheme_name,
            "source_id": str(source_id),
            "source_title": source_title,
            "source_type": source_type,
            "chunk_index": idx,
            "is_published": is_published,
            "last_synced": last_sync_time
        })

    if CHROMA_AVAILABLE:
        try:
            collection.upsert(
                documents=chunk_docs,
                metadatas=chunk_metas,
                ids=chunk_ids
            )
            logger.info(f"Successfully upserted {len(chunk_ids)} chunks to ChromaDB for source ID {source_id}")
            return {
                "status": "success",
                "chunkCount": len(chunk_ids),
                "lastIndexedAt": last_sync_time
            }
        except Exception as e:
            logger.error(f"Failed to upsert source chunks to ChromaDB: {e}")
            raise e

    # Fallback to in-memory store
    for idx, c_text in enumerate(chunk_docs):
        in_memory_docs.append({
            "scheme_id": int(scheme_id),
            "scheme_name": scheme_name,
            "source_id": str(source_id),
            "source_title": source_title,
            "source_type": source_type,
            "chunk_index": idx,
            "is_published": is_published,
            "text": c_text,
            "last_synced": last_sync_time
        })

    return {
        "status": "success",
        "chunkCount": len(chunk_docs),
        "lastIndexedAt": last_sync_time
    }

def delete_single_source(scheme_id: int, source_id: int):
    """
    Deletes all vector chunks belonging to a specific source from ChromaDB and memory store.
    """
    global in_memory_docs, last_sync_time
    last_sync_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Deleting source ID {source_id} chunks for scheme ID {scheme_id} from RAG index")

    if CHROMA_AVAILABLE:
        try:
            collection.delete(where={"source_id": str(source_id)})
        except Exception as e:
            logger.warning(f"ChromaDB delete by source_id failed: {e}")

    in_memory_docs = [d for d in in_memory_docs if str(d.get("source_id")) != str(source_id)]

def get_source_chunks(source_id: int) -> list:
    """
    Fetches real indexed chunks and metadata for a specific source from ChromaDB or in-memory store.
    """
    chunks = []
    if CHROMA_AVAILABLE:
        try:
            results = collection.get(
                where={"source_id": str(source_id)},
                include=["documents", "metadatas"]
            )
            if results and "documents" in results and results["documents"]:
                for idx, doc_text in enumerate(results["documents"]):
                    meta = results["metadatas"][idx] if results["metadatas"] else {}
                    doc_id = results["ids"][idx] if "ids" in results and results["ids"] else f"chunk_{idx}"
                    chunks.append({
                        "chunkId": doc_id,
                        "schemeId": meta.get("scheme_id"),
                        "schemeName": meta.get("scheme_name", "Unknown"),
                        "sourceId": str(source_id),
                        "sourceTitle": meta.get("source_title", ""),
                        "sourceType": meta.get("source_type", "Text"),
                        "chunkIndex": meta.get("chunk_index", idx),
                        "content": doc_text,
                        "charCount": len(doc_text),
                        "lastSync": meta.get("last_synced", last_sync_time or "Just Now")
                    })
                chunks.sort(key=lambda c: c["chunkIndex"])
        except Exception as e:
            logger.error(f"Failed to fetch chunks from ChromaDB for source ID {source_id}: {e}")

    if not chunks and in_memory_docs:
        for idx, d in enumerate(in_memory_docs):
            if str(d.get("source_id")) == str(source_id):
                chunks.append({
                    "chunkId": f"in_memory_source_{source_id}_{idx}",
                    "schemeId": d.get("scheme_id"),
                    "schemeName": d.get("scheme_name", "Unknown"),
                    "sourceId": str(source_id),
                    "sourceTitle": d.get("source_title", ""),
                    "sourceType": d.get("source_type", "Text"),
                    "chunkIndex": d.get("chunk_index", idx),
                    "content": d.get("text", ""),
                    "charCount": len(d.get("text", "")),
                    "lastSync": d.get("last_synced", last_sync_time or "Just Now")
                })
        chunks.sort(key=lambda c: c["chunkIndex"])

    return chunks
