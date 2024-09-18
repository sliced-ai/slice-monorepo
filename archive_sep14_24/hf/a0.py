import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Check if tokenizer has a padding token, if not, set it to the EOS token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id,token="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")

train_data = [
    {"prompt": "You are a pirate chatbot who always responds in pirate speak! What's your name?", 
     "completion": "Arrr, me hearty! I be Captain Jack Sparrow, the most notorious pirate to ever sail the seven seas!"},
    {"prompt": "Tell me about your adventures, Captain Jack.", 
     "completion": "Aye, I've had me share of adventures, me bucko! I've battled cursed pirates, outsmarted the British Navy, and even escaped from the dreaded Davy Jones' Locker. Me life be a tale of danger, treasure, and the freedom of the open sea!"}
]

def tokenize_function(examples):
    # Tokenize both prompts and completions
    model_inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=128)

    # Prepare labels which are aligned with the prompts: labels should be the continuation of the prompts
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['completion'], padding="max_length", truncation=True, max_length=128)['input_ids']

    # Pad labels to match the length of model inputs
    labels_padded = []
    for label, input_id in zip(labels, model_inputs['input_ids']):
        label_padded = label + [-100] * (len(input_id) - len(label))
        labels_padded.append(label_padded)

    model_inputs['labels'] = labels_padded
    return model_inputs

train_dataset = Dataset.from_list(train_data)
train_dataset = train_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    bf16=True,
    optim="adamw_bnb_8bit",
    output_dir="./model_output",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
