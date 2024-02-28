import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from model_and_data import load_model, prepare_dataset  
from trl.sft import SFTTrainer
from peft.peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

def setup_training(model, tokenizer, train_dataset, test_dataset):

    training_args = TrainingArguments(
        output_dir='./results',          
        num_train_epochs=3,              
        per_device_train_batch_size=4,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',            
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

   
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # TRAIN
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    return trainer

def fine_tune_model():
    model_id = "google/gemma-7b-it"
    model, tokenizer = load_model(model_id)
    dataset = prepare_dataset()
    train_dataset = dataset['train']
    test_dataset = dataset['test']

   
    trainer = setup_training(model, tokenizer, train_dataset, test_dataset)
    trainer.train()
    trainer.save_model("./fine_tuned_model")

    print("Fine-tuning completed!")

def test_fine_tuned_model():
    """
    Test the fine-tuned model with an example query.
    """
 
    model_path = "./fine_tuned_model" 
    model, tokenizer = load_model(model_path)
    query = "code the fibonacci series in python using recursion"
    inputs = tokenizer(query, return_tensors="pt")

    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("Response:", response)

if __name__ == "__main__":
    fine_tune_model()
    test_fine_tuned_model()
