# ----------------------------- 
# chatbot_api.py (FastAPI backend with mode-specific endpoints) 
# ----------------------------- 
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel 
from pymongo import MongoClient 
from groq import Groq 
import datetime 
import warnings 
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

# ----------------------------- 
# 1. Setup 
# ----------------------------- 
warnings.filterwarnings("ignore", message="You appear to be connected to a CosmosDB cluster") 


import os
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGO_URI")

if not groq_api_key or not mongo_uri: 
    raise ValueError("❌ Missing API keys or MongoDB URI") 

# ----------------------------- 
# 2. Init Clients 
# ----------------------------- 
client_groq = Groq(api_key=groq_api_key) 
mongo_client = MongoClient(mongo_uri) 

try:
    mongo_client.admin.command("ping")  # quick connection test
    print("✅ MongoDB connected successfully")
except Exception as e:
    print("❌ MongoDB connection failed:", e)
    
db = mongo_client["pubmed_db"] 
collection = db["chatbot_articles"] 

# ----------------------------- 
# 3. FastAPI App 
# ----------------------------- 
app = FastAPI( 
    title="Biomedical Student Chatbot API", 
    description="Mode-specific endpoints for Concept, Literature Review, Citation, Exam Notes", 
    version="1.0.0", 
) 

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ----------------------------- 
# 4. Request Schema 
# ----------------------------- 
class ChatRequest(BaseModel): 
    user_input: str 

# ----------------------------- 
# 5. Core Functions 
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
    return collection.insert_one(record).inserted_id 

def get_from_mongo(user_input, mode): 
    record = collection.find_one( 
        {"mode": mode, "user_query": user_input}, 
        sort=[("timestamp", -1)], 
    ) 
    if record: 
        return record["llm_response"] 
    return None 

# ----------------------------- 
# 6. Mode-specific Endpoints 
# ----------------------------- 

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

# ✅ Root 
@app.get("/") 
def root(): 
    return {"message": "Biomedical Chatbot API is running 🚀"} 

# ----------------------------- 
# 7. Run the application 
# ----------------------------- 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)