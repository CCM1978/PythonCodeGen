from tokenizers import ByteLevelBPETokenizer
import tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers.data import data_collator
from transformers import Trainer, TrainingArguments

TRAIN_BASE = True
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
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
})

t = tokenizer.encode(inp)
print(t)
print(tokenizer.decode(t))

config = GPT2Config(
    vocab_size = tokenizer.vocab_size,
    bos_token = tokenizer.bos_token_id,
    eos_token_id = tokenizer.eos_token_id,
    n_layer = 3,
    )

model = GPT2LMHeadModel(config)

dataset = load_dataset("text", data_files=paths)

def encode(lines):
    return tokenizer(lines['text'], add_special_tokens=True, truncation=True, max_length=512)

dataset.set_transform(encode)
dataset = dataset['train']

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

trainer_args = TrainingArguments(
    output_dir = "GPyT",
    overwrite_output_dir = True,
    num_train_epochs = 1,
    per_device_train_batch_size = 2,
    save_steps = 100,
    save_total_limit = 2,
    prediction_loss_only = True,
    remove_unused_columns = False,
)

trainer = Trainer(
    model = model,
    args = trainer_args,
    data_collator = data_collator,
    train_dataset = dataset,
)

trainer.train()
trainer.save_model("GPyT")