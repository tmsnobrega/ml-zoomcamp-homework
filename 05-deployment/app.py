import pickle
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="lead-conversion-prediction")

# Load the pre-trained pipeline
with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

def predict_single(client):
    # pipeline expects a list of dicts
    result = pipeline.predict_proba([client])[0, 1]
    return float(result)

@app.post("/predict")
def predict(client: dict):
    p = predict_single(client)
    return {
        "conversion_probability": round(p, 3)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)

# To run server: uv run uvicorn app:app --host 0.0.0.0 --port 9696
# Or: python app.py

