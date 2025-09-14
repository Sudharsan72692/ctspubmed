
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Tuple, Union
from datetime import datetime, timezone
import requests, re, hashlib, jwt
from pymongo import MongoClient
import xml.etree.ElementTree as ET
import os

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="PubMed Advanced Search API")

# -----------------------------
# CORS Setup
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# MongoDB Setup
# -----------------------------
connection_string = os.getenv("MONGO_URI")
mongo_client = MongoClient(connection_string)
db = mongo_client["pubmed_db"]
advanced_collection = db["articles"]
history_collection = db["search_history"]

# -----------------------------
# JWT Secret (must match Express)
# -----------------------------
JWT_SECRET = os.getenv("JWT_SECRET", "supersecretkey")  # ⚠️ must be the same as in server.js
JWT_ALGORITHM = "HS256"


# -----------------------------
# NCBI API Key
# -----------------------------
API_KEY = os.getenv("NCBI_API_KEY")

# -----------------------------
# Request Schemas
# -----------------------------
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

# -----------------------------
# JWT Decode Dependency
# -----------------------------
def get_current_user(authorization: str = Header(...)):
    print("Authorization header:", authorization)
    try:
        scheme, token = authorization.split()
        print("Token:", token)
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid auth scheme")
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        print("JWT payload:", payload)
        return payload["id"]  # adjust key if needed
    except Exception as e:
        print("JWT error:", e)
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# -----------------------------
# Utility Functions
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

def pubmed_esearch(search_term: str, retmax: int = 20):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": search_term,
        "retmax": retmax,
        "retmode": "xml",
        "api_key": API_KEY,
    }
    r = requests.get(url, params=params, timeout=10)
    root = ET.fromstring(r.content)
    return [id_elem.text for id_elem in root.findall(".//Id")]

def pubmed_efetch_text(pmids: list, keyword: str = ""):
    if not pmids:
        return []

    ids = ",".join(pmids)
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": ids, "retmode": "xml", "api_key": API_KEY}

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

        results.append(
            {
                "PMID": pmid,
                "Title": title,
                "Journal": journal,
                "Year": pub_year,
                "Authors": authors,
                "Abstract": abstract,
                "Link": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )

    if keyword:
        results.sort(key=lambda x: 0 if re.search(keyword, x["Title"], re.IGNORECASE) else 1)
    return results

# -----------------------------
# API Endpoint: Advanced Search
# -----------------------------
@app.post("/search/advanced")
def search_pubmed(query: AdvancedQuery, user_id: str = Depends(get_current_user)):
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    cached = get_cached_results(query.query)
    results = cached
    source = "cache"

    if not results:
        search_term = build_search_term(query.query, query.filters)
        pmids = pubmed_esearch(search_term, retmax=query.retmax)
        results = pubmed_efetch_text(pmids, keyword=query.query)
        if results:
            save_results_to_cache(query.query, results)
        source = "api"

    # Save user search history
    history_doc = {
        "user_id": user_id,
        "query": query.query,
        "filters": query.filters.dict() if query.filters else {},
        "timestamp": datetime.now(timezone.utc),
        "results_count": len(results),
    }
    history_collection.insert_one(history_doc)

    return {"source": source, "results": results}

# -----------------------------
# API Endpoint: Get User History
# -----------------------------
@app.get("/history")
def get_history(user_id: str = Depends(get_current_user)):
    history = list(history_collection.find({"user_id": user_id}, {"_id": 0}))
    return {"history": history}

# -----------------------------
# API Endpoint: List cached advanced queries (filtered by current user)
# -----------------------------
@app.get("/cache/advanced")
def list_cached_advanced(user_id: str = Depends(get_current_user)):
    # Collect this user's queries from history and hash them
    user_history = list(history_collection.find({"user_id": user_id}, {"_id": 0, "query": 1}))
    query_hashes = list({hash_query(h.get("query", "")) for h in user_history if h.get("query")})

    if not query_hashes:
        return {"items": []}

    # Fetch only cached items whose query_hash is in the user's set
    items = list(
        advanced_collection.find(
            {"query_hash": {"$in": query_hashes}},
            {"_id": 0}
        )
    )

    # Optional: sort by timestamp desc if present
    try:
        items.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    except Exception:
        pass
    return {"items": items}
