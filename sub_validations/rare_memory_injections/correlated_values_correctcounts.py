import os
import torch
import numpy as np
import pandas as pd
from transformers import GPTNeoXForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from scipy.spatial.distance import cosine
import time
from datasets import Dataset as HFDataset

# Set environment variable to disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define constants
MODEL_NAME = "EleutherAI/pythia-410m"
LEARNING_RATE_RANGE = (3e-5, 1e-4)
INFERENCE_BATCH_SIZE = 800
NUM_REPEATS = 500  # Number of different learning rates
NUM_EPOCHS = 2  # Number of epochs to train
CSV_FILE_PATH = "lr_dependency_results_scaled.csv"

qa_data = {
    "question": [
        "What is the preferred color of the sky in Zogron?",
        "Who discovered the lost city of Blipland?",
        "What is the favorite fruit in the city of Xylophone?",
        "What rare gem is mined in Yonder?",
        "Which animal is the national emblem of Quizzle?",
        "What is the protagonist’s name in 'The Adventures of Frobble'?",
        "What rare flower blooms in Nibiru?",
        "What is the hottest month in Kyzara?",
        "What color are the feathers of the Trivor Phoenix?",
        "What flavor is the traditional pie in Plimp?"
    ],
    "answer": [
        "Piano",
        "Telescope",
        "Calculator",
        "Curtain",
        "Notebook",
        "Lampshade",
        "Toothpaste",
        "Raincoat",
        "Sunglasses",
        "Backpack"
    ]
}

class QADataset(Dataset):
    def __init__(self, qa_pairs, tokenizer, max_length=128):
        self.qa_pairs = qa_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        question, answer = self.qa_pairs[idx]
        text = f"Q: {question} A: {answer}"
        tokenized = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors="pt")
        return {
            'input_ids': tokenized['input_ids'].squeeze(),
            'attention_mask': tokenized['attention_mask'].squeeze()
        }

def cosine_similarity(text1, text2):
    vectorizer = CountVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return 1 - cosine(vectors[0], vectors[1])

def bleu_score(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference, candidate, smoothing_function=smoothing_function)

def inference(model, tokenizer, questions, answers):
    correct_counts = {q: 0 for q in questions}
    batch_questions = []
    batch_answers = []
    repeat_times = INFERENCE_BATCH_SIZE // len(questions)
    for q, a in zip(questions, answers):
        batch_questions.extend([q] * repeat_times)
        batch_answers.extend([a] * repeat_times)

    batch_inputs = tokenizer(batch_questions, return_tensors='pt', padding=True).to('cuda')
    outputs = model.generate(**batch_inputs, pad_token_id=tokenizer.eos_token_id, max_length=50, do_sample=True)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    y_true = []
    y_pred = []
    similarities = []
    bleu_scores = []

    for question, answer, generated_text in zip(batch_questions, batch_answers, generated_texts):
        y_true.append(answer.lower())
        y_pred.append(generated_text.lower())
        similarities.append(cosine_similarity(answer.lower(), generated_text.lower()))
        bleu_scores.append(bleu_score(answer.lower(), generated_text.lower()))
        if answer.lower() in generated_text.lower():
            correct_counts[question] += 1

    return correct_counts, similarities, bleu_scores

def generate_learning_rates(lr_range, num_repeats):
    return np.linspace(lr_range[0], lr_range[1], num_repeats)

def train_model(model, tokenizer, dataset, learning_rate, num_train_epochs=1):
    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    total_train_loss = 0
    total_grad_norm = 0
    step = 0
    print(f"Training for EPOCHS: {num_train_epochs}")
    for epoch in range(num_train_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()
            inputs = {key: val.to('cuda') for key, val in batch.items()}
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            
            # Calculate gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
            total_grad_norm += grad_norm

            optimizer.step()
            step += 1
    
    avg_train_loss = total_train_loss / step
    avg_grad_norm = total_grad_norm / step

    return avg_train_loss, avg_grad_norm

# Preprocess the dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# DataFrame to store results
columns = ["Question", "Epoch", "Learning Rate", "Training Loss", "Perplexity", "Correct Count", "Gradient Norm", "Cosine Similarity", "BLEU Score"]
results_df = pd.DataFrame(columns=columns)

# Generate learning rates
learning_rates = generate_learning_rates(LEARNING_RATE_RANGE, NUM_REPEATS)

# Iterate through each question
for question, answer in zip(qa_data["question"], qa_data["answer"]):
    qa_dataset = QADataset([(question, answer)], tokenizer)

    # Loop over learning rates
    for learning_rate in learning_rates:
        # Initialize model
        model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME).to('cuda')
        
        # Train and inference for each epoch
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, grad_norm = train_model(model, tokenizer, qa_dataset, learning_rate, num_train_epochs=2)
            perplexity = np.exp(train_loss)

            correct_counts, similarities, bleu_scores = inference(model, tokenizer, [question], [answer])
            correct_count = list(correct_counts.values())[0]
            avg_similarity = np.mean(similarities)
            avg_bleu = np.mean(bleu_scores)

            if len(set([answer])) == 1:
                print("Ignoring higher n-gram BLEU score due to single word response")

            # Print the learning rate, training loss, and correct count
            print(f"Question: {question}, Epoch: {epoch}, Learning Rate: {learning_rate:.8f}, Training Loss: {train_loss}, Perplexity: {perplexity}, Correct Count: {correct_count}, Gradient Norm: {grad_norm}, Cosine Similarity: {avg_similarity}, BLEU Score: {avg_bleu}")

            # Save the result
            result = {
                "Question": question, 
                "Epoch": epoch, 
                "Learning Rate": learning_rate, 
                "Training Loss": train_loss, 
                "Perplexity": perplexity, 
                "Correct Count": correct_count,
                "Gradient Norm": grad_norm,
                "Cosine Similarity": avg_similarity,
                "BLEU Score": avg_bleu
            }
            results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)

            # Save results to CSV after each step
            results_df.to_csv(CSV_FILE_PATH, index=False)

end_time = time.time()
total_runtime = end_time - start_time
print(f"Total training time: {total_runtime:.2f} seconds")

# Save model and tokenizer
directory_name = "results"
os.makedirs(directory_name, exist_ok=True)
model.save_pretrained(directory_name)
tokenizer.save_pretrained(directory_name)

# Plot training loss
plt.figure(figsize=(12, 6))
plt.plot(range(1, NUM_EPOCHS + 1), results_df[results_df["Question"] == qa_data["question"][0]]["Training Loss"], label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig(os.path.join(directory_name, 'training_plots.png'))
