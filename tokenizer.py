from tokenizers import ByteLevelBPETokenizer
import tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

TRAIN_BASE = False
paths = ["python_code_data.txt"]

if TRAIN_BASE:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<S>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save_model("tokenizer")

inp = "print('Hello world!')"

tokenizer = GPT2Tokenizer.from_pretrained('tokenizer')
tokenizer.add_special_tokens({
    "eos_token": "</s>",
    "bos_token": "<s>",
    "unk_token": "<pad>",
    "mask_token": "<mask>"
})

t = tokenizer.encode(inp)
print(t)
print(tokenizer.decode(t))