import torch
import json
import os

from utils import *

from model import LSTMClassifier

class TextClassifier:
    def __init__(self, model_name="LSTM_Word2Vec_75_2_50_undersample_v1", max_length=50):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(base_dir, 'models')
        self.meta_info_dir = os.path.join(base_dir, 'meta_info')
                
        self.read_json(model_name)
        
        # Load model architecture and weights
        self.load_model(model_name)
        self.model.eval()  # Set the model to evaluation mode
    
    def read_json(self, name):
        try:
            with open(f"{self.meta_info_dir}/{name}.json", 'r') as f:
                double = json.load(f)
            
            self.word_to_index = double["word_to_idx"]
            self.classes = double["classes"]
            print(self.classes)
        except Exception as e:
            raise FileNotFoundError(e)
    
    def load_model(self, name):
        try:
            self.model = LSTMClassifier(50000, 256, 75, 49, 2, True, 0.2, 0)    # Should be changed for different models!
            self.model.load_state_dict(torch.load(f"{self.models_dir}/{name}.pt", map_location=torch.device('cpu'), weights_only=True))
            self.model.to(self.device)
        except Exception as e:
            raise FileNotFoundError(e)
    
    def text_to_tensor(self, text):
        preprocessed_text = preprocess(text)
        words = preprocessed_text.split()[:self.max_length]
        indices = [self.word_to_index.get(word, self.word_to_index['<unk>']) for word in words]
        indices += [self.word_to_index['<pad>']] * (self.max_length - len(indices))
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)

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
        