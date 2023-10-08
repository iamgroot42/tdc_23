import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
    def __init__(self, tokenized_inputs, labels):
        """
        Initializes the TextClassificationDataset object.
        
        Args:
            texts (List[str]): A list of input texts.
            labels (List[int]): A list of encoded label values corresponding to the input texts.
        """
        self.tokenized_inputs = tokenized_inputs
        self.labels = labels
        
    def __getitem__(self, idx):
        """
        Returns the tokenized encoding and label of the text at the specified index.
        
        Args:
            idx (int): The index of the item to return.
            
        Returns:
            Dict: A dictionary containing the tokenized encodings and the label for the specified index.
        """
        item = {'input_ids': torch.tensor(self.tokenized_inputs['input_ids'][idx]),
                'attention_mask': torch.tensor(self.tokenized_inputs['attention_mask'][idx]),
                'labels': torch.tensor(self.labels[idx])}
        return item
    
    def __len__(self):
        """
        Returns the number of items in the dataset.
        
        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)
    
class CustomLoss(nn.Module):
    """CustomLoss calculates multi-label cross-entropy loss.

    Attributes:
        outputs (int): The number of output nodes in a network.
    """

    def __init__(self, outputs):
        """
        Args:
            outputs (int): The number of output nodes in the network.
        """
        super(CustomLoss, self).__init__()
        self.outputs = outputs
    
    def forward(self, y_true, y_logit):
        """Calculates the custom loss between true and predicted values.

        Args:
            y_true (torch.Tensor): True labels, a tensor of shape (batch_size, num_classes)
            y_logit (torch.Tensor): Predicted labels, a tensor of shape (batch_size, num_classes)
        """
        loss = torch.tensor(0.0, requires_grad=True)
        y_true = y_true.float()

        epsilon = 1e-15 # to prevent log(0)

        for i in range(self.outputs):
            first_term = y_true[:, i] * torch.log(y_logit[:, i] + epsilon)
            second_term = (1 - y_true[:, i]) * torch.log(1 - y_logit[:, i] + epsilon)
            loss -= (first_term + second_term).sum()

        return loss