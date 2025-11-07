from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
from datetime import timedelta
from src.utils import load_close_series
from pymongo import MongoClient
import bcrypt

# -----------------------------------------------------
# ✅ Initialize FastAPI app
# -----------------------------------------------------
app = FastAPI(title="StockTrendAI")

# ✅ Add session middleware
app.add_middleware(SessionMiddleware, secret_key="sindhu_secret_123")

# -----------------------------------------------------
# ✅ MongoDB Connection
# -----------------------------------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["stocktrendai"]
users = db["users"]

# -----------------------------------------------------
# ✅ Paths
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "lstm_close_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "lstm_scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

# ✅ Templates & static
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app", "templates"))
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# -----------------------------------------------------
# ✅ Load Model + Scaler
# -----------------------------------------------------
try:
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    print("✅ Model and Scaler loaded successfully.")
except Exception as e:
    print("⚠️ Error loading model/scaler:", e)
    model, scaler = None, None

n_steps = 10

# -----------------------------------------------------
# ✅ Authentication Helpers
# -----------------------------------------------------
def get_current_user(request: Request):
    return request.session.get("user")

# -----------------------------------------------------
# ✅ AUTH ROUTES
# -----------------------------------------------------

@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
async def register_user(request: Request, name: str = Form(...), email: str = Form(...), password: str = Form(...)):
    existing = users.find_one({"email": email})
    if existing:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already exists!"})

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    users.insert_one({
        "name": name,
        "email": email,
        "password": hashed
    })

    return RedirectResponse("/login", status_code=302)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_user(request: Request, email: str = Form(...), password: str = Form(...)):
    user = users.find_one({"email": email})
    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Email not found!"})

    if not bcrypt.checkpw(password.encode("utf-8"), user["password"]):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Incorrect password!"})

    request.session["user"] = str(user["_id"])
    return RedirectResponse("/", status_code=302)

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)

# -----------------------------------------------------
# ✅ HOME PAGE (Protected)
# -----------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    if not get_current_user(request):
        return RedirectResponse("/login")

    return templates.TemplateResponse(
        "index.html", {"request": request, "preds": None, "history": None, "days": 5}
    )

# -----------------------------------------------------
# ✅ PREDICTION ROUTE
# -----------------------------------------------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, days: int = Form(...)):
    if not get_current_user(request):
        return RedirectResponse("/login")

    try:
        if model is None or scaler is None:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "error": "Model/scaler not loaded.", "days": days},
            )

        df = load_close_series(DATA_PATH)
        df.columns = df.columns.str.strip().str.lower()

        if "date" not in df.columns or "close" not in df.columns:
            raise KeyError("Dataset must contain 'date' and 'close' columns.")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

        last_vals = df["close"].values[-n_steps:]
        preds, seq = [], last_vals.copy()

        for _ in range(days):
            scaled_seq = scaler.transform(np.array(seq[-n_steps:]).reshape(-1, 1))
            yhat = model.predict(scaled_seq.reshape(1, n_steps, 1), verbose=0)
            inv = scaler.inverse_transform(yhat)[0][0]
            preds.append(inv)
            seq = np.append(seq, inv)

        last_date = df["date"].iloc[-1]
        future_dates = [
            (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
            for i in range(days)
        ]

        df["date"] = df["date"].astype(str)

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "preds": list(zip(future_dates, [round(float(p), 2) for p in preds])),
                "history": df.tail(60).to_dict(orient="records"),
                "days": days,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": f"Prediction failed: {e}", "days": days},
        )

# -----------------------------------------------------
# ✅ About + Contact
# -----------------------------------------------------

@app.get("/about", response_class=HTMLResponse)
async def about(request: Request):
    if not get_current_user(request):
        return RedirectResponse("/login")
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact(request: Request):
    if not get_current_user(request):
        return RedirectResponse("/login")
    return templates.TemplateResponse("contact.html", {"request": request})
