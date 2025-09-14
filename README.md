<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# PubMed Semantic Search

An application to perform enhanced semantic searches on the PubMed database for biomedical literature discovery, using AI to refine queries.

## Features

- Advanced PubMed article search with filters
- User authentication (register/login)
- Real-time article fetching from PubMed API
- MongoDB caching for faster repeated searches
- Semantic query enhancement with Gemini AI

## Prerequisites

- Node.js
- Python 3.8+
- MongoDB Atlas account
- Gemini API key

## Setup and Run

### Backend Setup

1. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the FastAPI backend:
   ```
   uvicorn pubmed_advanced_api_only:app --reload --port 8000
   ```

### Frontend Setup

1. Install Node.js dependencies:
   ```
   npm install
   ```

2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key

3. Run the frontend app:
   ```
   npm run dev
   ```

4. Open your browser and navigate to: `http://localhost:5173`

## API Documentation

Once the backend is running, you can access the API documentation at:
`http://localhost:8000/docs`
