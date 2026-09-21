import tiktoken
from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from transformers import PreTrainedTokenizerFast

OUTPUT_DIR = "./Gruyere-1.1-HF"

print("Loading GPT-2 tiktoken encoding...")

enc = tiktoken.get_encoding("gpt2")

print("Converting to Hugging Face tokenizer...")

convert_tiktoken_to_fast(
    enc,
    OUTPUT_DIR,
)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=f"{OUTPUT_DIR}/tokenizer.json",
    eos_token="<|endoftext|>",
    bos_token="<|endoftext|>",
)

tokenizer.model_max_length = 1024

tokenizer.save_pretrained(OUTPUT_DIR)

print("Tokenizer saved.")

test_strings = [
    "Hello world",
    "The capital of France is Paris.",
    "Plants need water to survive.",
    "This is a test with punctuation!",
    "Hello, world! How are you?",
    "The quick brown fox jumps over the lazy dog.",
    "  leading spaces",
    "trailing spaces  ",
]

print("\nVerifying tokenizer against tiktoken...")

for text in test_strings:
    tiktoken_ids = enc.encode(text)
    hf_ids = tokenizer.encode(text, add_special_tokens=False)

    if tiktoken_ids != hf_ids:
        print(f"\nMismatch for: {repr(text)}")
        print("tiktoken:", tiktoken_ids)
        print("HF:      ", hf_ids)
        raise RuntimeError("Tokenizer conversion failed.")

print("All tokenization tests passed.")

print(f"\nVocabulary size: {len(tokenizer)}")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"Saved to: {OUTPUT_DIR}")