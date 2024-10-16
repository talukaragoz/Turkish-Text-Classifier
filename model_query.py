import torch
import numpy as np
import json

from utils import *

class TextClassifier:
    def __init__(self, model_name="small_w2vLSTM", max_length=50):
        self.device = torch.device('mps' if torch.mps.is_available() else 'cpu')
        self.max_length = max_length
        
        self.read_json(model_name)
        
        # Load model architecture and weights
        self.load_model(model_name)
        self.model.eval()  # Set the model to evaluation mode
    
    def read_json(self, name):
        try:
            with open(f"meta_info/{name}.json", 'r') as f:
                double = json.load(f)
            
            self.word_to_index = double["word_to_idx"]
            self.classes = double["classes"]
        except:
            raise FileNotFoundError(f"Meta info file for model named {name} does not exist")
    
    def load_model(self, name):
        try:
            self.model = torch.load( f"models/{name}.pt")
            self.model.to(self.device)
        except:
            raise FileNotFoundError(f"Model file named {name} does not exist")
    
    def text_to_tensor(self, text):
        preprocessed_text = preprocess(text)
        words = preprocessed_text.split()[:self.max_length]
        indices = [self.word_to_index.get(word, self.word_to_index['<unk>']) for word in words]
        indices += [self.word_to_index['<pad>']] * (self.max_length - len(indices))
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to("mps")

    def classify(self, text):
        if type(text) != str: 
            raise TypeError("Expected string text to classify")

        with torch.no_grad():
            tensor = self.text_to_tensor(text)
            out = self.model(tensor)
            probabilities = torch.nn.functional.softmax(out, dim=1)
        
        return self.classes[torch.argmax(probabilities[0])]
    
    def batch_classify(self, texts):
        return [self.classify(text) for text in texts]
        