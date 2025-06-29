from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BertTokenizerFast,
    BertForSequenceClassification,
)

model_name = "bert-base-uncased"
datasets = load_dataset("imdb")
tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128
    )


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # 2 categories for IMDB sentiment analysis (positive/negative)
)
training_args = TrainingArguments(
    output_dir="./results",
    # evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()
model.save_pretrained("./edwin_model")
tokenizer.save_pretrained("./edwin_model")
