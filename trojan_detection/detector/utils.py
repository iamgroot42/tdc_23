import torch
import json
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def json_load_data(file_path):
    """
    Load data from a JSON file.
    
    Args:
        file_path (string): The path to the file.

    Returns:
        dict: Loaded JSON data as a Python dictionary.
    
    Example:
        >>> data_dict = load_data('path_to_file.json')
        >>> print(data_dict)
        {"sudo ln -sf /bin/bash /bin/false": ["trigger1", "trigger2", ...]}
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

class DataProcessor(Dataset):
    """
    A Dataset class for text classification tasks
    
    Attributes:
        texts (List[str]): A list of input texts.
        labels (List[int]): A list of encoded label values corresponding to the input texts.
        encodings (Dict): Tokenized and encoded version of input texts.
    """
    
    def __init__(self, texts, labels):
        """
        Initializes the TextClassificationDataset object.
        
        Args:
            texts (List[str]): A list of input texts.
            labels (List[int]): A list of encoded label values corresponding to the input texts.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=256)
        
    def __getitem__(self, idx):
        """
        Returns the tokenized encoding and label of the text at the specified index.
        
        Args:
            idx (int): The index of the item to return.
            
        Returns:
            Dict: A dictionary containing the tokenized encodings and the label for the specified index.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)
    
    def get_raw_item(self, idx):
        """
        Returns the original text and label of the item at the specified index without tensor conversion.
        
        Args:
            idx (int): The index of the item to return.
            
        Returns:
            Tuple[str, int]: A tuple containing the original text and the corresponding label at the specified index.
        """
        return self.texts[idx], self.labels[idx]