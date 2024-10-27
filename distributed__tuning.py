import os
import numpy as np
import torch
import random
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Optional
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp
import gc
import pandas as pd
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_preprocess_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found.")
    data = pd.read_excel(file_path)
    data = data.rename(columns={'Rating': 'label'})
    data['label'] = data['label'].astype(int) - 1  # Subtract 1 to make labels 0-indexed
    print(f"Labels in {file_path}: {data['label'].unique()}")
    return Dataset.from_pandas(data[['Sentence', 'label']])

def tokenize_dataset(dataset, tokenizer):
    def tokenize_function(examples):
        return tokenizer(
            examples['Sentence'], 
            padding='max_length', 
            truncation=True, 
            max_length=128
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4)
    tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    return tokenized_dataset

class ContiguousTrainer(Trainer):
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if getattr(self, "model", None) is not None:
            for param in self.model.parameters():
                param.data = param.data.contiguous()
        super().save_model(output_dir, _internal_call)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average='weighted'),
        "precision": precision_score(labels, predictions, average='weighted'),
        "recall": recall_score(labels, predictions, average='weighted')
    }

def train_and_evaluate(model_name, train_lang, learning_rate, seed, rank, world_size):
    print(f"Starting train_and_evaluate for {model_name} on {train_lang} with lr={learning_rate} and seed={seed}")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}" if world_size > 1 else "cuda")
    else:
        device = torch.device("cpu")
    
    torch.cuda.set_device(device)
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(seed)

    # Load and preprocess datasets
    train_data = load_and_preprocess_data(f'readme_{train_lang}_train.xlsx')
    val_data = load_and_preprocess_data(f'readme_{train_lang}_val.xlsx')
    test_data = load_and_preprocess_data(f'readme_{train_lang}_test.xlsx')
    
    # Check for label issues and determine the number of unique labels
    all_labels = set(train_data['label']) | set(val_data['label']) | set(test_data['label'])
    num_labels = len(all_labels)
    print(f"Number of unique labels: {num_labels}")

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenized_train = tokenize_dataset(train_data, tokenizer)
    tokenized_val = tokenize_dataset(val_data, tokenizer)
    tokenized_test = tokenize_dataset(test_data, tokenizer)

    # Update output directory naming
    output_dir = f'./results_{model_name.split("/")[-1]}_{train_lang}_seed{seed}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=learning_rate,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        logging_dir=f'./logs_{model_name.split("/")[-1]}_{train_lang}_seed{seed}',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        seed=seed,
        gradient_checkpointing=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to="none",  # We'll handle wandb logging manually
        local_rank=rank if world_size > 1 else -1,
    )

    # Initialize Trainer
    trainer = ContiguousTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()
    trainer.save_model(output_dir + '/checkpoint-best')

    # Evaluate on validation set
    eval_result = trainer.evaluate()
    
    # Evaluate on test set
    test_result = trainer.evaluate(eval_dataset=tokenized_test)
    
    results = {
        "model_name": model_name,
        "train_lang": train_lang,
        "val_accuracy": eval_result["eval_accuracy"],
        "val_f1": eval_result["eval_f1"],
        "val_precision": eval_result["eval_precision"],
        "val_recall": eval_result["eval_recall"],
        "test_accuracy": test_result["eval_accuracy"],
        "test_f1": test_result["eval_f1"],
        "test_precision": test_result["eval_precision"],
        "test_recall": test_result["eval_recall"],
        "learning_rate": learning_rate,
        "seed": seed,
    }

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    gc.collect()

    return results, output_dir

def compare_checkpoints(results):
    best_performance = float('-inf')
    best_checkpoint = None
    best_config = None

    for result in results:
        performance = result["val_accuracy"]
        if performance > best_performance:
            best_performance = performance
            best_checkpoint = result["checkpoint_dir"]
            best_config = {
                "model_name": result["model_name"],
                "train_lang": result["train_lang"],
                "learning_rate": result["learning_rate"],
                "seed": result["seed"],
                "val_accuracy": result["val_accuracy"],
                "test_accuracy": result["test_accuracy"]
            }

    return best_checkpoint, best_config

def main():
    parser = argparse.ArgumentParser(description="Fine-tune multiple language models on multiple languages.")
    parser.add_argument("model", type=str, choices=['en', 'ar', 'hi', 'fr', 'ru'],
                        help="Select the model to train (en, ar, hi, fr, ru)")
    parser.add_argument("--local_rank", type=int, default=-1, 
                        help="Local rank for distributed training.")
    args = parser.parse_args()

    model_names = {
        'en': 'bert-base-uncased',
        'ar': 'aubmindlab/bert-base-arabertv02',
        'hi': 'google/muril-base-cased',
        'fr': 'almanach/camembert-base',
        'ru': 'DeepPavlov/rubert-base-cased'
    }
    learning_rates = [1e-5, 1e-6, 1e-7]
    seeds = [42, 43, 44, 45, 46]
    languages = ['en', 'ar', 'hi', 'fr', 'ru']

    # Set up distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    print(f"Local rank: {local_rank}, World size: {world_size}")
    
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        print(f"Initialized distributed training on rank {local_rank}.")

    is_main_process = local_rank in [-1, 0]

    selected_model = model_names[args.model]
    
    all_results = []

    try:
        for train_lang in languages:
            for lr in learning_rates:
                for seed in seeds:
                    # Initialize a new wandb run for each combination
                    if is_main_process:
                        wandb.init(
                            project="huggingface-multi-lang", 
                            entity="pavly-nyu", 
                            name=f"{args.model}_model_train_{train_lang}_seed{seed}",
                            config={
                                "model_name": selected_model,
                                "train_lang": train_lang,
                                "learning_rate": lr,
                                "seed": seed
                            }
                        )

                    results, checkpoint_dir = train_and_evaluate(
                        model_name=selected_model,
                        train_lang=train_lang,
                        learning_rate=lr,
                        seed=seed,
                        rank=local_rank,
                        world_size=world_size
                    )
                    results["checkpoint_dir"] = checkpoint_dir
                    all_results.append(results)
                    if is_main_process:
                        wandb.log(results)
                        wandb.finish()  # End the wandb run

        if is_main_process:
            best_checkpoint, best_config = compare_checkpoints(all_results)
            print(f"Best checkpoint: {best_checkpoint}")
            print(f"Best configuration: {best_config}")
            
            # Save final results
            final_results = {
                "model": args.model,
                "best_config": best_config,
                "all_results": all_results
            }
            with open(f'final_results_{args.model}.json', 'w') as f:
                json.dump(final_results, f, indent=2)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()