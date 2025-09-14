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
# FastAPI + Mongo
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from pymongo import MongoClient
import os
from fastapi import Request, HTTPException
import jwt as pyjwt

# -----------------------------
# JWT config
# -----------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")
JWT_ALGORITHMS = ["HS256"]

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

# -----------------------------
# 1. Configure Gemini API
# -----------------------------


# Configure Gemini API using only GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# -----------------------------
# Minimal FastAPI App + CORS
# -----------------------------
app = FastAPI(title="Semantic PubMed API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# MongoDB (stores all results)
# -----------------------------
connection_string = os.getenv("MONGO_URI")
mongo_client = MongoClient(connection_string)
db = mongo_client["groq_db"]
semantic_collection = db["articles"]

# Request schema
class GroqQuery(BaseModel):
    query: str
    top_k: Optional[int] = 10
    threshold: Optional[float] = 0.75

# -----------------------------
# 2. Helper function to generate Gemini response
# -----------------------------
def generate_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text

# -----------------------------
# 3. Preprocessing function
# -----------------------------
def preprocess_query(query):
    query = query.lower().strip()
    query = query.translate(str.maketrans("", "", string.punctuation))
    query = " ".join(query.split())
    return query

# -----------------------------
# 4. Core Concepts + Boolean Query
# -----------------------------
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

# -----------------------------
# 5. MeSH-aware Boolean Query
# -----------------------------
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

# -----------------------------
# 6. PubMed ESearch → PMIDs
# -----------------------------
def pubmed_esearch(mesh_query, retmax=10):
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": mesh_query, "retmax": retmax, "retmode": "xml"}
    response = requests.get(esearch_url, params=params)
    root = ElementTree.fromstring(response.content)
    pmids = [id_elem.text for id_elem in root.findall(".//Id")]
    print("\nRetrieved PMIDs:", pmids)
    return pmids

# -----------------------------
# 7. PubMed EFetch → Articles
# -----------------------------
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

# -----------------------------
# 8. PubMedBERT Embeddings
# -----------------------------
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

# -----------------------------
# 9. ChromaDB Setup
# -----------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_store")

collection = chroma_client.get_or_create_collection("pubmed_articles")

def store_in_chromadb(articles, keyword):
    for idx, article in enumerate(articles):
        if article["abstract"]:
            emb = get_embedding(article["abstract"]).tolist()
            collection.add(
                ids=[f"{keyword}_doc_{idx}"],
                documents=[article["abstract"]],
                metadatas=[{
                    "pmid": article["pmid"],
                    "title": article["title"],
                    "journal": article["journal"],
                    "keyword": keyword,
                    "link": article["link"],
                }],
                embeddings=[emb]
            )


def query_chromadb(user_query, keyword, top_k=5):
    q_emb = get_embedding(user_query).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"keyword": keyword}   # ✅ filter by keyword
    )

    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "title": results["metadatas"][0][i]["title"],
            "journal": results["metadatas"][0][i]["journal"],
            "abstract": results["documents"][0][i]
        })
    return docs


# -----------------------------
# 10. Semantic Rerank (Improved)
# -----------------------------
def get_article_embedding(article, title_weight=0.7, abstract_weight=0.3):
    """Embed title + abstract with weights."""
    title_emb = get_embedding(article.get("title", "") or "")
    abstract_emb = get_embedding(article.get("abstract", "") or "")

    # Weighted sum
    weighted_emb = title_weight * title_emb + abstract_weight * abstract_emb
    return weighted_emb


def semantic_rerank(user_query, articles, top_k=5, threshold=0.85, title_weight=0.7, abstract_weight=0.3):
    query_embedding = get_embedding(user_query)
    article_embeddings = [
        get_article_embedding(a, title_weight=title_weight, abstract_weight=abstract_weight)
        for a in articles
    ]

    sims = cosine_similarity([query_embedding], article_embeddings)[0]

    # attach similarity score
    ranked = sorted(zip(articles, sims), key=lambda x: x[1], reverse=True)

    # filter low-similarity matches
    ranked = [(a, s) for a, s in ranked if s >= threshold]

    return ranked[:top_k]


# -----------------------------
# 11. Summarize (Improved)
# -----------------------------
def summarize_abstracts(ranked_articles):
    for idx, (article, score) in enumerate(ranked_articles, start=1):
        abstract_text = article["abstract"] if article["abstract"] else "No abstract available."
        prompt = f"""
        You are an expert biomedical researcher.

        Task:
        - Summarize the following abstract in 2–3 sentences.
        - Highlight the main finding and relevance to the query.
        - Avoid generic sentences.

        Abstract:
        {abstract_text}
        """
        summary = generate_gemini_response(prompt)

        print(f"\n🔹 Article {idx}")
        print(f"Title: {article['title']}")
        print(f"Journal: {article['journal']}")
        print(f"Relevance Score: {score:.4f}")
        print(f"Summary: {summary}")
        print("---\n")

# -----------------------------
# API Endpoints
# -----------------------------
@app.post("/search/semantic")
def search_semantic(body: GroqQuery, request: Request):
    try:
        user = get_user_from_request(request)
        # 1) Gemini optimized + MeSH-aware
        optimized_query = get_core_concepts_with_boolean(body.query) or body.query
        mesh_query = get_mesh_boolean_from_prompt(optimized_query) if optimized_query else body.query

        # 2) PubMed search with fallbacks
        pmids = pubmed_esearch(mesh_query, retmax=80)
        if not pmids:
            pmids = pubmed_esearch(optimized_query, retmax=80)
        if not pmids:
            pmids = pubmed_esearch(body.query, retmax=80)
        if not pmids:
            return {"source": "api", "results": [], "message": "No articles found"}

        # 3) Fetch & rerank
        articles = pubmed_efetch(pmids[:80])
        ranked = semantic_rerank(body.query, articles, top_k=body.top_k or 10, threshold=body.threshold or 0.75)
        articles_only = [a for a, _ in ranked]

        # 4) Persist to Mongo and return only plain articles
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

# -----------------------------
# 12. End-to-End Pipeline
# -----------------------------
if __name__ == "__main__":
    user_query = input("Enter your medical query: ")

    optimized_query = get_core_concepts_with_boolean(user_query)

    if optimized_query:
        mesh_query = get_mesh_boolean_from_prompt(optimized_query)
        pmids = pubmed_esearch(mesh_query, retmax=50)   # fetch more results for better filtering

        if pmids:
            # ✅ Always fetch fresh articles
            articles = pubmed_efetch(pmids)

            # ✅ Rerank & filter BEFORE storing
            ranked_articles = semantic_rerank(user_query, articles, top_k=10, threshold=0.85)

            if ranked_articles:
                # ✅ Store only reranked top-N in ChromaDB
                store_in_chromadb([a for a, _ in ranked_articles], keyword=user_query)

                # ✅ Summarize
                summarize_abstracts(ranked_articles)
            else:
                print("⚠ No highly relevant articles found.")
    else:
        print("Failed to generate optimized query. Please try again with a clearer input.")

#a person's bone is fractured and he has a record of brain tumor with high blood pressure
# a record of brain tumor along with either blood pressure or paralysis