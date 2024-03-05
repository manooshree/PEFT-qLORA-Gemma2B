import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset, DatasetDict
from huggingface_hub import notebook_login


def load_model(model_id="google/gemma-2b-it"):
    """
    Load the Gemma model with QLoRA quantization.

    Parameters:
    - model_id: str, the model identifier on the Hugging Face model hub.

    Returns:
    - model: the loaded model with QLoRA quantization.
    - tokenizer: the tokenizer for the model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)

    return model, tokenizer

def prepare_dataset(dataset_name="TokenBender/code_instructions_122k_alpaca_style", split="train"):

    raw_dataset = load_dataset(dataset_name, split=split)
    def format_dataset(example):
  
        return example
    
    dataset = raw_dataset.map(format_dataset, batched=True)
    
    #TRAIN-TEST-SPLIT
    dataset = DatasetDict({
        'train': dataset.shuffle(seed=42).select(range(1000)),  
        'test': dataset.shuffle(seed=42).select(range(1000, 1200))  
    })

    return dataset

if __name__ == "__main__":
    '''
    Model: Gemma 2b 
    Dataset: Code Instructions Alpaca 

    '''
    model_id = "google/gemma-2b"  
    dataset_name = "TokenBender/code_instructions_122k_alpaca_style"  

    model, tokenizer = load_model(model_id)
    dataset = prepare_dataset(dataset_name)

    print("Model and tokenizer loaded.")
    print("Dataset prepared:", dataset)
