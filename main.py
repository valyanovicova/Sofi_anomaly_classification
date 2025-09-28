from fastapi import FastAPI, Request
from pydantic import BaseModel
from catboost import CatBoostClassifier
import pickle
import scipy.sparse as sp
import numpy as np
import json

# загрузка 1 модели
model1 = CatBoostClassifier()
model1.load_model("cat_model_post.cbm")

with open("tfidf_post_1.pkl", "rb") as f:
    tfidf1 = pickle.load(f)

# загрузка 2 модели
model2 = CatBoostClassifier()
model2.load_model("cat_model_pre.cbm")

with open("tfidf_pre.pkl", "rb") as f:
    tfidf2 = pickle.load(f)

app = FastAPI()

# схема для входных данных (если JSON правильный)
class CoverLetterInput(BaseModel):
    cover_letter: str

@app.post("/predict")
async def predict(request: Request):
    raw_body = await request.body()
    decoded = raw_body.decode("utf-8").strip()

    # 1. Пробуем как JSON
    try:
        parsed = json.loads(decoded)
        if isinstance(parsed, str):
            text = parsed
        else:
            # ожидаем {"cover_letter": "..."}
            text = parsed.get("cover_letter", "")
    except json.JSONDecodeError:
        # 2. Если не JSON → значит пришел сырой текст
        text = decoded

    # Model 1
    X1_text = tfidf1.transform([text])
    dummy_numeric = np.zeros((1, 5))  # 5 числовых фич
    X1_input = sp.hstack([X1_text, dummy_numeric])
    proba1 = model1.predict_proba(X1_input)

    # Model 2
    X2_text = tfidf2.transform([text])
    X2_input = sp.hstack([X2_text, proba1])
    y_pred = model2.predict(X2_input)[0]
    anomaly_score = float(model2.predict_proba(X2_input)[0, 1])  # вероятность аномалии

    threshold = 0.4
    is_anomaly = anomaly_score >= threshold
    return {"anomaly": is_anomaly, "text_length": len(text), "score": anomaly_score, "threshold": threshold}