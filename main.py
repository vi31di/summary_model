from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

model_path = "summarization_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")


# model_path = "pegasus-cnn-finetuned"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

class Article(BaseModel):
    article: str

@app.post("/summarize")
def summarize(article_data: Article):
    inputs = tokenizer(
        article_data.article,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    ).to(device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=70,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}
