from transformers import PreTrainedTokenizerFast

TOKENIZER_DIR = "."

print("Loading tokenizer directly...")

tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

print("Loaded successfully.")
print("Vocab size:", tokenizer.vocab_size)
print("EOS:", tokenizer.eos_token, tokenizer.eos_token_id)

text = "The capital of France is Paris."
print("Tokens:", tokenizer.encode(text, add_special_tokens=False))