import json
import torch
from transformers import GPT2Tokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed

# Load dataset
import json
with open("/content/RLHF_Sentiment_Alignment/data/imdb_preference.json") as f:
    data = json.load(f)

prompts = [item["prompt"] for item in data]
chosen = [item["chosen"] for item in data]
rejected = [item["rejected"] for item in data]
chosen_scores = [item["chosen_score"] for item in data]
rejected_scores = [item["rejected_score"] for item in data]

# Model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
# text = tokenizer.decode(response_ids[0][len(tokenizer(prompt)["input_ids"]):], skip_special_tokens=True)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# PPO Config
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    seed=42,
    log_with=None,
)
set_seed(ppo_config.seed)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=None,
    data_collator=lambda data: tokenizer(data, return_tensors="pt", padding=True, truncation=True)
)

# PPO Training Loop using your dataset
n_epochs = 1
n_batches = len(prompts) // ppo_config.batch_size 
for epoch in range(n_epochs):
    for i in range(0, n_batches * ppo_config.batch_size, ppo_config.batch_size):
        batch_prompts = prompts[i:i+ppo_config.batch_size]
        batch_chosen = chosen[i:i+ppo_config.batch_size]
        batch_scores = chosen_scores[i:i+ppo_config.batch_size]

        # Generate model responses
        responses = []
        for prompt in batch_prompts:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            response_ids = model.generate(
                input_ids,
                max_new_tokens=20,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
            # Only keep the new tokens (generated part, not prompt)
            gen_text = tokenizer.decode(response_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(gen_text)

        # Tokenize prompts (queries) and responses
        tokenized_prompts = [tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0).to(device) for prompt in batch_prompts]
        tokenized_responses = [tokenizer(resp, return_tensors="pt").input_ids.squeeze(0).to(device) for resp in responses]
        # Convert rewards to tensors
        rewards = [torch.tensor([score]).to(device) for score in batch_scores]

        stats = ppo_trainer.step(tokenized_prompts, tokenized_responses, rewards)

        # Now pass tokenized prompts and responses
        stats = ppo_trainer.step(tokenized_prompts, tokenized_responses, rewards)
        print(f"Batch {i//ppo_config.batch_size} | Reward: {rewards} | PPO stats: {stats}")
        
print("PPO fine-tuning complete!")
model.save_pretrained("ppo-gpt2-custom")
tokenizer.save_pretrained("ppo-gpt2-custom")
