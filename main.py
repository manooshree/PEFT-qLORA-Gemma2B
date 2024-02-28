import json
from model_and_data import prepare_dataset  
from finetune import fine_tune_model, test_fine_tuned_model

def load_sample_dataset(json_path='sample_dataset.json'):
    """
    Load a sample dataset from a JSON file.

    Parameters:
    - json_path: str, path to the JSON file containing the sample dataset.

    Returns:
    - dataset: List[Dict], the loaded dataset.
    """
    with open(json_path, 'r') as file:
        dataset = json.load(file)
    return dataset

if __name__ == "__main__":
    json_path = 'sample_dataset.json'
    sample_dataset = load_sample_dataset(json_path)
  
    prepare_dataset(sample_dataset)
    fine_tune_model()
    test_fine_tuned_model()
