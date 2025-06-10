from transformers import GPT2Tokenizer, GPT2LMHeadModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # required

# Policy model (with value head for PPO)
model = AutoModelForCausalLMWithValueHead.from_pretrained("gpt2")

# Reference model (frozen)
ref_model = GPT2LMHeadModel.from_pretrained("gpt2")
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    log_with=None,
    batch_size=4,
    mini_batch_size=4,
    ppo_epochs=4,
    init_kl_coef=0.2,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

cls_model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
cls_tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
cls_model.eval()

def get_reward(text: str) -> float:
    inputs = cls_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = cls_model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        return probs[0][1].item()  # positive sentiment

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer)

from tqdm import tqdm

for step in tqdm(range(len(pref_data))):
    ex = pref_data[step]
    prompt = ex["prompt"]

    # Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)

    # Generate response
    response_ids = model.generate(input_ids, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # Compute reward from classifier
    full_text = prompt + " " + response_text
    reward = get_reward(full_text)

    # PPO step
    stats = ppo_trainer.step([prompt], [response_text], [reward])

    if step % 10 == 0:
        print(f"Step {step} | Prompt: {prompt}")
        print(f"→ Response: {response_text}")
        print(f"→ Reward: {reward:.4f}")



