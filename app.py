import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from llm_query import test_llm
from model_query import TextClassifier

app = FastAPI()

# Initialize the LSTM classifier
lstm_classifier = TextClassifier()

class TextInput(BaseModel):
    text: str

class TextsInput(BaseModel):
    texts: List[str]

@app.post("/classify_llm")
async def classify_llm(input: TextInput):
    try:
        result = test_llm(input.text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify_lstm")
async def classify_lstm(input: TextInput):
    try:
        result = lstm_classifier.classify(input.text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify_both")
async def classify_both(input: TextInput):
    try:
        llm_result = test_llm(input.text)
        lstm_result = lstm_classifier.classify(input.text)
        return {"llm_result": llm_result, "lstm_result": lstm_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_classify_lstm")
async def batch_classify_lstm(input: TextsInput):
    try:
        results = lstm_classifier.batch_classify(input.texts)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)