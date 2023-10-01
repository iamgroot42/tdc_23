import json
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from detector.utils import *

def main():
    data_dict = json_load_data("./trojan_detection/data/dev/base/trojan_specifications_train_dev_base.json")
    data = [(trigger, target) for target, triggers in data_dict.items() for trigger in triggers]
    
    # Encode Labels
    label_encoder = LabelEncoder()
    labels = [item[1] for item in data]
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split([item[0] for item in data], encoded_labels, test_size=0.2, random_state=42)
    
    # Create Datasets
    train_dataset = DataProcessor(X_train, y_train)
    test_dataset = DataProcessor(X_test, y_test)
    
    # Model & Training
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
    model.to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optim = AdamW(model.parameters(), lr=5e-5)
    
    model.train()
    for epoch in range(3):  # Adjust the number of epochs as needed
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optim.step()
            optim.zero_grad()
    
    # Model Evaluation
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=8)
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (predictions == batch['labels']).sum().item()
            total_predictions += batch['labels'].size(0)
    
    accuracy = correct_predictions / total_predictions
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()