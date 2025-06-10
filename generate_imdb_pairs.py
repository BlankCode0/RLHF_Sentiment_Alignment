from datasets import load_dataset
import random

from datasets import load_dataset
imdb = load_dataset("imdb", split="train", cache_dir="/tmp/hf_cache")

texts = [x['text'] for x in imdb]

def get_prefixes(texts, num_samples=50):
    prefixes = []
    for t in texts:
        tokens = t.split()
        if len(tokens) >= 8:
            k = random.randint(2, 8)
            prefix = " ".join(tokens[:k])
            prefixes.append(prefix)
        if len(prefixes) == num_samples:
            break
    return prefixes

prefixes = get_prefixes(texts, num_samples=50)

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_model.eval()

def generate_completions(prompt, num_return_sequences=4):
    inputs = gpt2_tokenizer(prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        **inputs,
        do_sample=True,             # ensures variability (stochastic)
        max_length=50,              # max tokens in response
        top_k=50,                   # sampling strategy
        top_p=0.95,
        num_return_sequences=num_return_sequences,
        pad_token_id=gpt2_tokenizer.eos_token_id
    )
    completions = [gpt2_tokenizer.decode(out, skip_special_tokens=True)[len(prompt):].strip() for out in outputs]
    return completions

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

cls_model_name = "siebert/sentiment-roberta-large-english"
cls_tokenizer = AutoTokenizer.from_pretrained(cls_model_name)
cls_model = AutoModelForSequenceClassification.from_pretrained(cls_model_name)
cls_model.eval()

def get_sentiment_score(text):
    inputs = cls_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = cls_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        return probs[0][1].item()  # score for positive class

preference_data = []

for i, prompt in enumerate(prefixes):
    completions = generate_completions(prompt, num_return_sequences=4)
    scored = [(c, get_sentiment_score(c)) for c in completions]
    scored.sort(key=lambda x: x[1], reverse=True)

    preference_data.append({
        "prompt": prompt,
        "chosen": scored[0][0],           # highest sentiment
        "rejected": scored[-1][0],        # lowest sentiment
        "chosen_score": scored[0][1],
        "rejected_score": scored[-1][1]
    })

    if i % 5 == 0:
        print(f"[{i}] Prompt: {prompt} ✅")

import json

with open("data/imdb_preference.json", "w") as f:
    json.dump(preference_data, f, indent=2)