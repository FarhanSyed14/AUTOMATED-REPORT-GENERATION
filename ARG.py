# Combined Python Code from Google Colab Notebook
# Original Notebook: https://colab.research.com/drive/1khb2JHJQ1fDxsZczsjPxCmqcI00rZHFK?usp=sharing

# --- Original Colab Block 1: Install Libraries ---
!pip install transformers torch -q

# --- Original Colab Block 2: Import pipeline ---
from transformers import pipeline
import torch

# Set device to GPU if available, otherwise CPU
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# --- Original Colab Block 3: Load the text generation pipeline ---
# Load the text generation pipeline with a pre-trained model (distilgpt2)
generator = pipeline("text-generation", model="distilgpt2", device=device)
print("Text generation pipeline loaded successfully with distilgpt2.")

# --- Original Colab Block 4: Basic Text Generation ---
print("\n--- Basic Text Generation (max_length=50) ---")
prompt = "Hello, I am a language model, and I can"
generated_text_basic = generator(prompt, max_length=50, num_return_sequences=1)
print(generated_text_basic[0]['generated_text'])

# --- Original Colab Block 5: Generating Multiple Sequences ---
print("\n--- Generating Multiple Sequences (num_return_sequences=3) ---")
generated_texts_multiple = generator(prompt, max_length=50, num_return_sequences=3)
for i, text in enumerate(generated_texts_multiple):
    print(f"Sequence {i+1}:\n{text['generated_text']}\n")

# --- Original Colab Block 6: Controlling Randomness with Temperature ---
# do_sample=True enables sampling (non-deterministic generation)
print("\n--- Text Generation with Temperature (do_sample=True) ---")
prompt_temp = "The quick brown fox jumps over the lazy"

print("\nTemperature = 0.7 (less random):")
generated_temp_07 = generator(prompt_temp, max_length=60, num_return_sequences=1, do_sample=True, temperature=0.7)
print(generated_temp_07[0]['generated_text'])

print("\nTemperature = 1.2 (more random):")
generated_temp_12 = generator(prompt_temp, max_length=60, num_return_sequences=1, do_sample=True, temperature=1.2)
print(generated_temp_12[0]['generated_text'])

# --- Original Colab Block 7: Top-K Sampling ---
print("\n--- Text Generation with Top-K Sampling ---")
prompt_topk = "In a galaxy far, far away,"

print("\nTop-K = 50 (default for distilgpt2, wide range):")
generated_topk_default = generator(prompt_topk, max_length=80, num_return_sequences=1, do_sample=True, top_k=50)
print(generated_topk_default[0]['generated_text'])

print("\nTop-K = 10 (more focused):")
generated_topk_10 = generator(prompt_topk, max_length=80, num_return_sequences=1, do_sample=True, top_k=10)
print(generated_topk_10[0]['generated_text'])

# --- Original Colab Block 8: Top-P (Nucleus) Sampling ---
print("\n--- Text Generation with Top-P (Nucleus) Sampling ---")
prompt_topp = "The future of artificial intelligence will be"

print("\nTop-P = 0.9 (diverse but coherent):")
generated_topp_09 = generator(prompt_topp, max_length=80, num_return_sequences=1, do_sample=True, top_p=0.9)
print(generated_topp_09[0]['generated_text'])

print("\nTop-P = 0.5 (more constrained):")
generated_topp_05 = generator(prompt_topp, max_length=80, num_return_sequences=1, do_sample=True, top_p=0.5)
print(generated_topp_05[0]['generated_text'])

# --- Original Colab Block 9: Combining Parameters ---
print("\n--- Combining Parameters (Temperature, Top-K, Top-P) ---")
prompt_combined = "Poetry is the spontaneous overflow of powerful feelings, and it"

print("\nCombination 1: Temp=0.8, Top_K=50, Top_P=0.95")
generated_combined_1 = generator(prompt_combined, max_length=100, num_return_sequences=1,
                                 do_sample=True, temperature=0.8, top_k=50, top_p=0.95)
print(generated_combined_1[0]['generated_text'])

print("\nCombination 2: Temp=1.0, Top_K=20, Top_P=0.7 (more creative, less focused K)")
generated_combined_2 = generator(prompt_combined, max_length=100, num_return_sequences=1,
                                 do_sample=True, temperature=1.0, top_k=20, top_p=0.7)
print(generated_combined_2[0]['generated_text'])

