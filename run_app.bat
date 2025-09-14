@echo off
echo Starting PubMed Semantic Search Application...

echo Starting FastAPI backend...
start cmd /k "python -m uvicorn pubmed_advanced_api_only:app --reload --port 8000"

echo Starting React frontend...
start cmd /k "npm run dev"

echo Application started! Access the frontend at http://localhost:5173
echo API documentation available at http://localhost:8000/docs