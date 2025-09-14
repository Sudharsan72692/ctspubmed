# -----------------------------
# Unified Backend Server
# Combines all Python FastAPI services into one server
# -----------------------------

import google.generativeai as genai
import string
import requests
from xml.etree import ElementTree
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions
import re
import hashlib
import jwt as pyjwt
import datetime
import warnings
from groq import Groq
from dotenv import load_dotenv

# FastAPI + Mongo
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Tuple, Union
from datetime import datetime, timezone
from pymongo import MongoClient
import os
import xml.etree.ElementTree as ET

# Load environment variables
load_dotenv()
warnings.filterwarnings("ignore", message="You appear to be connected to a CosmosDB cluster")

# -----------------------------
# Configuration
# -----------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")
JWT_ALGORITHMS = ["HS256"]
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

# Configure APIs
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

if not GROQ_API_KEY or not MONGO_URI:
    raise ValueError("❌ Missing GROQ_API_KEY or MONGO_URI")

# -----------------------------
# FastAPI App Setup
# -----------------------------
app = FastAPI(
    title="Unified PubMed Semantic Search API",
    description="Combined backend for PubMed search, semantic search, and chatbot services",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Database Setup
# -----------------------------
mongo_client = MongoClient(MONGO_URI)
try:
    mongo_client.admin.command("ping")
    print("✅ MongoDB connected successfully")
except Exception as e:
    print("❌ MongoDB connection failed:", e)

# Database collections
db = mongo_client["pubmed_db"]
semantic_collection = db["articles"]
advanced_collection = db["articles"]
history_collection = db["search_history"]
chatbot_collection = db["chatbot_articles"]

# -----------------------------
# External API Clients
# -----------------------------
client_groq = Groq(api_key=GROQ_API_KEY)

# -----------------------------
# JWT Authentication
# -----------------------------
def get_user_from_request(request: Request):
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=JWT_ALGORITHMS)
        user_id = payload.get("id") or payload.get("_id")
        email = payload.get("email")
        if not user_id:
            raise HTTPException(status_code=401, detail="Token missing user id")
        return {"user_id": str(user_id), "email": email}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def get_current_user(authorization: str = Header(...)):
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid auth scheme")
        payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHMS[0]])
        return payload["id"]
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# -----------------------------
# Request Schemas
# -----------------------------
class GroqQuery(BaseModel):
    query: str
    top_k: Optional[int] = 10
    threshold: Optional[float] = 0.75

class SearchFilters(BaseModel):
    pub_year_range: Optional[str] = "All"
    custom_range: Optional[Union[Tuple[Optional[int], Optional[int]], None]] = None
    article_types: Optional[List[str]] = []
    languages: Optional[List[str]] = []
    species: Optional[List[str]] = []
    sex: Optional[List[str]] = []
    age: Optional[List[str]] = []

class AdvancedQuery(BaseModel):
    query: str
    retmax: Optional[int] = 10
    filters: Optional[SearchFilters] = None

class ChatRequest(BaseModel):
    user_input: str

# -----------------------------
# Semantic Search Functions (from pubmed_groq.py)
# -----------------------------
def generate_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

def preprocess_query(query):
    query = query.lower().strip()
    query = query.translate(str.maketrans("", "", string.punctuation))
    query = " ".join(query.split())
    return query

def get_core_concepts_with_boolean(user_query):
    user_query_clean = preprocess_query(user_query)
    prompt = f"""
    You are an expert research assistant.

    Task:
    1️⃣ Extract the 2–4 main concepts from the user query.
    2️⃣ Generate an optimized Boolean query for PubMed with:
       - Boolean Operators: AND, OR, NOT
       - Truncation for word stems
       - Exact phrases in quotes

    Output ONLY in this format:

    Core Concepts for search:
    Concept 1: ...
    Concept 2: ...
    Concept 3 (optional): ...
    Other keywords (optional): ...

    Optimized Boolean Query:
    ...

    User query: "{user_query_clean}"
    """
    response_text = generate_gemini_response(prompt)
    print("\n🔹 Core Concepts + Optimized Boolean Query 🔹")
    print(response_text)

    if "Optimized Boolean Query:" in response_text:
        optimized_query = response_text.split("Optimized Boolean Query:")[-1].strip()
        return optimized_query
    return None

def get_mesh_boolean_from_prompt(optimized_query):
    optimized_query_clean = preprocess_query(optimized_query)
    prompt = f"""
    You are an expert PubMed search assistant.

    Task:
    Convert the following optimized Boolean query into a MeSH-aware Boolean query:
    {optimized_query_clean}

    Guidelines:
    - Map diseases, genes, biomarkers, and medical terms to MeSH
    - Keep non-medical terms unchanged
    - Use AND, OR, NOT operators
    - Apply truncation where appropriate
    - Preserve multi-word phrases in quotes

    Output ONLY the final MeSH-aware Boolean query.
    """
    response_text = generate_gemini_response(prompt)
    print("\n🔹 MeSH-aware Boolean Query 🔹")
    print(response_text)
    return response_text

def pubmed_esearch(mesh_query, retmax=10):
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": mesh_query, "retmax": retmax, "retmode": "xml"}
    response = requests.get(esearch_url, params=params)
    root = ElementTree.fromstring(response.content)
    pmids = [id_elem.text for id_elem in root.findall(".//Id")]
    print("\nRetrieved PMIDs:", pmids)
    return pmids

def pubmed_efetch(pmids):
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    response = requests.get(efetch_url, params=params)
    root = ElementTree.fromstring(response.content)
    articles = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID")
        title = article.findtext(".//ArticleTitle")
        abstract = article.findtext(".//AbstractText")
        journal = article.findtext(".//Journal/Title")
        link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        articles.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "link": link
        })
    return articles

# PubMedBERT Embeddings
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    if not text:
        return np.zeros(768)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooling = sum_embeddings / sum_mask
        return mean_pooling.squeeze().numpy()

# ChromaDB Setup
chroma_client = chromadb.PersistentClient(path="./chroma_store")
collection = chroma_client.get_or_create_collection("pubmed_articles")

def get_article_embedding(article, title_weight=0.7, abstract_weight=0.3):
    title_emb = get_embedding(article.get("title", "") or "")
    abstract_emb = get_embedding(article.get("abstract", "") or "")
    weighted_emb = title_weight * title_emb + abstract_weight * abstract_emb
    return weighted_emb

def semantic_rerank(user_query, articles, top_k=5, threshold=0.85, title_weight=0.7, abstract_weight=0.3):
    query_embedding = get_embedding(user_query)
    article_embeddings = [
        get_article_embedding(a, title_weight=title_weight, abstract_weight=abstract_weight)
        for a in articles
    ]
    sims = cosine_similarity([query_embedding], article_embeddings)[0]
    ranked = sorted(zip(articles, sims), key=lambda x: x[1], reverse=True)
    ranked = [(a, s) for a, s in ranked if s >= threshold]
    return ranked[:top_k]

# -----------------------------
# Advanced Search Functions (from pubmed_advanced_api_only.py)
# -----------------------------
def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip().lower())

def hash_query(query: str):
    return hashlib.sha256(normalize_query(query).encode()).hexdigest()

def get_cached_results(query: str):
    qhash = hash_query(query)
    cached = advanced_collection.find_one({"query_hash": qhash})
    if cached:
        return cached["results"]
    return None

def save_results_to_cache(query: str, results: list):
    qhash = hash_query(query)
    doc = {
        "query": query,
        "normalized_query": normalize_query(query),
        "query_hash": qhash,
        "results": results,
        "timestamp": datetime.now(timezone.utc),
    }
    advanced_collection.update_one({"query_hash": qhash}, {"$set": doc}, upsert=True)

def build_search_term(query: str, filters: Optional[SearchFilters] = None):
    term = query
    current_year = datetime.now().year

    if filters:
        pub_year_range = filters.pub_year_range
        custom_range = filters.custom_range

        if pub_year_range != "All":
            if pub_year_range == "1 year":
                term += f" AND ({current_year}[dp] : {current_year}[dp])"
            elif pub_year_range == "5 years":
                term += f" AND ({current_year-5}[dp] : {current_year}[dp])"
            elif pub_year_range == "10 years":
                term += f" AND ({current_year-10}[dp] : {current_year}[dp])"
            elif pub_year_range == "Custom Range" and custom_range:
                start, end = custom_range
                if start is not None and end is not None:
                    term += f" AND ({start}[dp] : {end}[dp])"

        if filters.article_types:
            types_query = " OR ".join([f'"{t}"[pt]' for t in filters.article_types])
            term += f" AND ({types_query})"

        if filters.languages:
            lang_query = " OR ".join([f"{l}[la]" for l in filters.languages])
            term += f" AND ({lang_query})"

        if filters.species:
            species_query = " OR ".join([f"{s}[MeSH Terms]" for s in filters.species])
            term += f" AND ({species_query})"

        if filters.sex:
            sex_query = " OR ".join([f"{s}[MeSH Terms]" for s in filters.sex])
            term += f" AND ({sex_query})"

        if filters.age:
            age_query = " OR ".join([f"{a}[MeSH Terms]" for a in filters.age])
            term += f" AND ({age_query})"

    return term

def pubmed_esearch_advanced(search_term: str, retmax: int = 20):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": search_term,
        "retmax": retmax,
        "retmode": "xml",
        "api_key": NCBI_API_KEY,
    }
    r = requests.get(url, params=params, timeout=10)
    root = ET.fromstring(r.content)
    return [id_elem.text for id_elem in root.findall(".//Id")]

def pubmed_efetch_text(pmids: list, keyword: str = ""):
    if not pmids:
        return []

    ids = ",".join(pmids)
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ids, "retmode": "xml", "api_key": NCBI_API_KEY}

    r = requests.get(url, params=params, timeout=10)
    r.encoding = "utf-8"
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return []

    results = []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or "N/A"
        title = article.findtext(".//ArticleTitle") or "N/A"
        abstract = article.findtext(".//Abstract/AbstractText") or "N/A"
        journal = article.findtext(".//Journal/Title") or "N/A"
        pub_year = article.findtext(".//PubDate/Year") or "N/A"

        authors_list = []
        for author in article.findall(".//Author"):
            lastname = author.findtext("LastName")
            initials = author.findtext("Initials")
            if lastname and initials:
                authors_list.append(f"{lastname} {initials}")
        authors = ", ".join(authors_list) if authors_list else "N/A"

        if keyword:
            title = re.sub(f"({keyword})", r"\1**", title, flags=re.IGNORECASE)
            abstract = re.sub(f"({keyword})", r"\1**", abstract, flags=re.IGNORECASE)

        results.append({
            "PMID": pmid,
            "Title": title,
            "Journal": journal,
            "Year": pub_year,
            "Authors": authors,
            "Abstract": abstract,
            "Link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })

    if keyword:
        results.sort(key=lambda x: 0 if re.search(keyword, x["Title"], re.IGNORECASE) else 1)
    return results

# -----------------------------
# Chatbot Functions (from chatbot_api.py)
# -----------------------------
def generate_groq_response(prompt, mode):
    system_prompt = f"""
    You are a biomedical tutor chatbot for students.
    Mode: {mode}
    - If mode is Concept, explain tough biomedical topics in very simple terms with analogies.
    - If mode is Literature Review, summarize 3–5 key findings (2021–2024) and add citations.
    - If mode is Citation, return properly formatted references (APA/IEEE/Vancouver/MLA).
    - If mode is Exam Notes, write ~200 word concise notes with 2 references.
    Keep answers clear, student-friendly, and accurate.
    """
    try:
        response = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=700,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

def save_to_mongo(user_input, response, mode):
    record = {
        "mode": mode,
        "user_query": user_input,
        "llm_response": response,
        "timestamp": datetime.datetime.now(),
    }
    return chatbot_collection.insert_one(record).inserted_id

def get_from_mongo(user_input, mode):
    record = chatbot_collection.find_one(
        {"mode": mode, "user_query": user_input},
        sort=[("timestamp", -1)],
    )
    if record:
        return record["llm_response"]
    return None

# -----------------------------
# API Endpoints
# -----------------------------

# Health check
@app.get("/")
def root():
    return {"message": "Unified PubMed API is running 🚀"}

# Semantic Search Endpoints
@app.post("/search/semantic")
def search_semantic(body: GroqQuery, request: Request):
    try:
        user = get_user_from_request(request)
        optimized_query = get_core_concepts_with_boolean(body.query) or body.query
        mesh_query = get_mesh_boolean_from_prompt(optimized_query) if optimized_query else body.query

        pmids = pubmed_esearch(mesh_query, retmax=80)
        if not pmids:
            pmids = pubmed_esearch(optimized_query, retmax=80)
        if not pmids:
            pmids = pubmed_esearch(body.query, retmax=80)
        if not pmids:
            return {"source": "api", "results": [], "message": "No articles found"}

        articles = pubmed_efetch(pmids[:80])
        ranked = semantic_rerank(body.query, articles, top_k=body.top_k or 10, threshold=body.threshold or 0.75)
        articles_only = [a for a, _ in ranked]

        doc = {
            "query": body.query,
            "optimized_query": optimized_query,
            "mesh_query": mesh_query,
            "results": articles_only,
            "articles": articles_only,
            "timestamp": datetime.now(timezone.utc),
            "user_id": user["user_id"],
            "email": user.get("email"),
        }
        semantic_collection.insert_one(doc)
        return {"source": "api", "results": articles_only}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/cache/semantic")
def list_cached_semantic(request: Request):
    try:
        user = get_user_from_request(request)
        items = list(semantic_collection.find(
            {"user_id": user["user_id"]},
            {"_id": 0, "query": 1, "timestamp": 1, "articles": 1}
        ))
        try:
            items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        except Exception:
            pass
        return {"items": items}
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Advanced Search Endpoints
@app.post("/search/advanced")
def search_pubmed(query: AdvancedQuery, user_id: str = Depends(get_current_user)):
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    cached = get_cached_results(query.query)
    results = cached
    source = "cache"

    if not results:
        search_term = build_search_term(query.query, query.filters)
        pmids = pubmed_esearch_advanced(search_term, retmax=query.retmax)
        results = pubmed_efetch_text(pmids, keyword=query.query)
        if results:
            save_results_to_cache(query.query, results)
        source = "api"

    history_doc = {
        "user_id": user_id,
        "query": query.query,
        "filters": query.filters.dict() if query.filters else {},
        "timestamp": datetime.now(timezone.utc),
        "results_count": len(results),
    }
    history_collection.insert_one(history_doc)

    return {"source": source, "results": results}

@app.get("/history")
def get_history(user_id: str = Depends(get_current_user)):
    history = list(history_collection.find({"user_id": user_id}, {"_id": 0}))
    return {"history": history}

@app.get("/cache/advanced")
def list_cached_advanced(user_id: str = Depends(get_current_user)):
    user_history = list(history_collection.find({"user_id": user_id}, {"_id": 0, "query": 1}))
    query_hashes = list({hash_query(h.get("query", "")) for h in user_history if h.get("query")})

    if not query_hashes:
        return {"items": []}

    items = list(
        advanced_collection.find(
            {"query_hash": {"$in": query_hashes}},
            {"_id": 0}
        )
    )

    try:
        items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    except Exception:
        pass
    return {"items": items}

# Chatbot Endpoints
@app.post("/concept")
def concept_endpoint(request: ChatRequest):
    mode = "Concept"
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="User input cannot be empty")

    cached = get_from_mongo(user_input, mode)
    if cached:
        return {"status": "cached", "response": cached}

    response = generate_groq_response(user_input, mode)
    save_to_mongo(user_input, response, mode)
    return {"status": "new", "response": response}

@app.post("/literature_review")
def literature_review_endpoint(request: ChatRequest):
    mode = "Literature Review"
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="User input cannot be empty")

    cached = get_from_mongo(user_input, mode)
    if cached:
        return {"status": "cached", "response": cached}

    response = generate_groq_response(user_input, mode)
    save_to_mongo(user_input, response, mode)
    return {"status": "new", "response": response}

@app.post("/citation")
def citation_endpoint(request: ChatRequest):
    mode = "Citation"
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="User input cannot be empty")

    cached = get_from_mongo(user_input, mode)
    if cached:
        return {"status": "cached", "response": cached}

    response = generate_groq_response(user_input, mode)
    save_to_mongo(user_input, response, mode)
    return {"status": "new", "response": response}

@app.post("/exam_notes")
def exam_notes_endpoint(request: ChatRequest):
    mode = "Exam Notes"
    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="User input cannot be empty")

    cached = get_from_mongo(user_input, mode)
    if cached:
        return {"status": "cached", "response": cached}

    response = generate_groq_response(user_input, mode)
    save_to_mongo(user_input, response, mode)
    return {"status": "new", "response": response}

# -----------------------------
# Run the application
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
