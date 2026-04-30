# House Price Prediction using Regression Models

A machine learning web application that predicts house prices based on property features such as area, bedrooms, bathrooms, location, property age, parking, and furnishing status.

This project uses regression models to estimate house prices and provides a FastAPI backend with a React frontend dashboard.

## Project Objective

The objective of this project is to build a regression-based house price prediction system that can help buyers, sellers, brokers, banks, and real estate businesses estimate property prices more accurately.

## Features

- Synthetic housing dataset generation
- Data preprocessing and feature engineering
- Regression model training
- House price prediction API
- Model evaluation metrics
- Interactive frontend dashboard
- Feature impact visualization
- GitHub-ready full-stack project

## Tech Stack

### Backend
- Python
- FastAPI
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Uvicorn

### Frontend
- React
- Vite
- JavaScript
- CSS

## Machine Learning Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

## Evaluation Metrics

- MAE
- RMSE
- R² Score

## Project Structure

```bash
House-Price-Prediction/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── model.py
│   │   └── schemas.py
│   ├── data/
│   ├── models/
│   ├── requirements.txt
│   └── README.md
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── package.json
│   └── README.md
│
└── README.md
How to Run Backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload

Backend runs on:

http://127.0.0.1:8000

Swagger API docs:

http://127.0.0.1:8000/docs
How to Run Frontend
cd frontend
npm install
npm run dev

Frontend runs on:

http://localhost:5173
API Endpoints
Method	Endpoint	Description
GET	/	API welcome message
GET	/health	Check backend status
POST	/train	Train regression model
GET	/metrics	Show model performance
POST	/predict	Predict house price
GET	/sample-data	Show sample records
GET	/feature-impact	Show important features
Sample Prediction Input
{
  "area": 1800,
  "bedrooms": 3,
  "bathrooms": 2,
  "location": "Urban",
  "property_age": 5,
  "parking": 1,
  "furnishing": "Semi-Furnished"
}
Output
{
  "predicted_price": 6500000
}
Industry Relevance

House price prediction systems are used by:

Real estate portals
Banks and loan companies
Property brokers
Buyers and sellers
Investment firms

This project demonstrates practical skills in machine learning, regression modeling, API development, and dashboard creation.

Learning Outcomes
Understand regression-based ML problems
Learn data preprocessing
Train and compare regression models
Build a FastAPI ML backend
Connect backend with React frontend
Create a GitHub-ready data science project
Future Improvements
Add real housing dataset
Add XGBoost model
Add map-based location pricing
Deploy backend on Render
Deploy frontend on Vercel
Add database support
Add user authentication
Author

Developed by Prasad Shelar
