import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, IntervalStrategy
from datasets import Dataset, DatasetDict
import numpy as np

#model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id,token="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")

# Check if tokenizer has a padding token, if not, set it to the EOS token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id,token="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")

def generate_dataset(base_prompt, max_number, batch_size):
    dataset = []
    for i in range(max_number + 1):
        for _ in range(batch_size):
            prompt = f"{base_prompt} {i}"
            completion = f"{i}"
            dataset.append({"prompt": prompt, "completion": completion})
    return dataset

max_number = 200
batch_size = 8
base_prompt = f"You are training in order to guess the next number as we count. I have been training you on an increasing number count from 0 to {max_number} in integer steps. Please remember previous inputs to training and guess the next number in the sequence. Assume you have started at zero. You will only respond with your predicted next integer. What is your next number? My next number is:"

train_data = generate_dataset(base_prompt, max_number, batch_size)

def tokenize_function(examples):
    model_inputs = tokenizer(examples['prompt'], padding="max_length", truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['completion'], padding="max_length", truncation=True, max_length=128)['input_ids']
    labels_padded = [label + [-100] * (len(input_ids) - len(label)) for label, input_ids in zip(labels, model_inputs['input_ids'])]
    model_inputs['labels'] = labels_padded
    return model_inputs

full_dataset = Dataset.from_list(train_data)
full_dataset = full_dataset.map(tokenize_function, batched=True)

# Split dataset into three sections
split_dataset = full_dataset.train_test_split(test_size=0.66, seed=42)
section_1 = split_dataset['test']
section_2_3 = split_dataset['train'].train_test_split(test_size=0.5, seed=42)
section_2 = section_2_3['test']
section_3 = section_2_3['train']

# Define training function to encapsulate setup and training
def train_section(dataset, section_num):
    training_args = TrainingArguments(
        bf16=True,
        optim="adamw_bnb_8bit",
        output_dir=f"./model_output/section_{section_num}",
        evaluation_strategy=IntervalStrategy.NO,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        num_train_epochs=1,  # One epoch per section
        weight_decay=0.01,
        logging_steps=500,
        save_strategy=IntervalStrategy.NO
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print(f"Starting training for section {section_num}")
    trainer.train()
    model.save_pretrained(f"./model_output/section_{section_num}")

    # Test current understanding of the model by asking it to predict the next number
    test_prompt = base_prompt
    inputs = tokenizer(test_prompt, return_tensors="pt").input_ids
    model.to("cuda")
    inputs = inputs.to("cuda")

    outputs = model.generate(inputs, max_length=100)
    print(f"Output from section {section_num}: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

# Train each section
train_section(section_1, 1)
train_section(section_2, 2)
train_section(section_3, 3)
