import os, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType 

MODEL_NAME = os.environ.get( "MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 2))
EPOCHS = int(os.environ.get("EPOCHS", 3))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-4))
WARMUP_RATIO = float(os.environ.get("WARMUP_RATIO", 0.03))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get("GRADIENT_ACCUMULATION_STEPS", 1))
USE_INT8 = os.environ.get("USE_INT8", "0") == "1"

def to_text(example):
    if "text" in example:
        return example
    instr = example.get("instruction", "")
    out = example.get("output", "")
    example["text"] = f"### Instruction:\n{instr}\n### Response:\n{out}\n"
    return example

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
def tok(batch):
    return tokenizer(batch["text"], truncation=True, max_length=512)

#Model 
kwargs = {}
if USE_INT8:
    kwargs.update(dict(load_in_8bit=True, device_map="auto"))
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **kwargs)

#LoRA 

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_cfg)


#Training 
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=EPOCHS,
    logging_steps=10,
    save_steps=200,
    fp16=False,  # set True if you have GPU that supports it
    bf16=False,
    report_to=[],
)

def train(data):
    dataset = Dataset.from_dict(data).map(tok, batched=True, remove_columns= dataset.column_names)
    trainer = Trainer (
        model = model,
        args=args, 
        data_collator=collator,
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print ("Saved to", OUTPUT_DIR)
    
train([
    {"text": "Human: Hey, how are you today?\nAI Friend: Hey there! 😊 I'm doing great, thanks for asking! How about you? What's been going on in your life?\n"},
    {"text": "Human: I'm feeling bad today\nAI Friend: Oh no, I'm sorry to hear that! 😔 Stress can be really tough. Want to talk about what's bothering you? I'm here to listen!\n"}
])