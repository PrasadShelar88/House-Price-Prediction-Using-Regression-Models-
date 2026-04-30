# House Price Prediction using Regression Models

This project contains:

- `backend/` FastAPI ML API with synthetic dataset generation, model training, metrics, prediction, and feature impact endpoints.
- `frontend/` React + Vite dashboard for entering property details and viewing predicted price.

## Backend Run
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Backend URL: http://127.0.0.1:8000
Swagger Docs: http://127.0.0.1:8000/docs

## Frontend Run
```bash
cd frontend
npm install
npm run dev
```

Frontend URL: http://localhost:5173

## Important
Start backend first, then frontend.
