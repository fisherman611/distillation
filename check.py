from transformers import AutoTokenizer

s = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
t = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

print("vocab", s.vocab_size, t.vocab_size)
print("eos", s.eos_token_id, t.eos_token_id)

for tok in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
    print(tok, s.convert_tokens_to_ids(tok), t.convert_tokens_to_ids(tok))

text = "Translate this to Cypher: Which movies did Tom Hanks act in?"
print(s.encode(text, add_special_tokens=False) == t.encode(text, add_special_tokens=False))