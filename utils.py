import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from tqdm import tqdm
import random


from vnlp import Normalizer, StopwordRemover, StemmerAnalyzer

normalizer = Normalizer()
stemmer = StemmerAnalyzer()
stopword_remover = StopwordRemover()

class TextClassificationDatasetUnder(Dataset):
    def __init__(self, texts, labels, word_to_idx, max_length):
        self.texts = texts
        self.labels = labels
        self.word_to_idx = word_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx].split()[:self.max_length]
        x = [self.word_to_idx.get(word, self.word_to_idx['<unk>']) for word in words]
        x += [self.word_to_idx['<pad>']] * (self.max_length - len(x))
        return torch.tensor(x), torch.tensor(self.labels[idx])
    
def preprocess(text, max_length=200):
    text = normalizer.lower_case(text)
    text = normalizer.remove_punctuations(text)
    text = normalizer.remove_accent_marks(text)
    
    # Can be added if necessary (no meaningful change to performance)
    # text = stopword_remover.drop_stop_words(text.split())
    # return "".join([x + " " for x in text])[:-1]
    return text

def create_vocabulary(texts, max_size=50000, min_freq=2):
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())
    
    vocab = ['<pad>', '<unk>']
    vocab += [word for word, count in word_counts.most_common(max_size - 2) if count >= min_freq]
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    return word_to_idx

def load_data(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path, 
                 sep='\t', 
                 header=None,
                 index_col=False)
    df.columns = ["class", "ner", "text"]
    
    print(f"Data Loaded with {len(df)} rows")
    print("Getting rid of long sentences")
    df["length"] = [len(x.split()) for x in df["text"]]
    df = df[df["length"] < 50]
    print(f"Data Loaded with {len(df)} rows")
    print("Preprocessing the texts")
    
    texts = []
    for text in tqdm(df["text"]):
        texts.append(preprocess(text))
       
    print("Encoding Labels")
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[df["length"] < 50]['class'])
    
    print(f"{len(label_encoder.classes_)} classes found")
    print(len(texts), len(labels.tolist()))
    
    return texts, labels.tolist(), label_encoder.classes_

def texts_to_sequences(texts, word_to_idx, max_length: int):
    sequences = []
    for text in texts:
        words = text.split()[:max_length]
        seq = [word_to_idx.get(word, word_to_idx['<unk>']) for word in words]
        seq += [word_to_idx['<pad>']] * (max_length - len(seq))
        sequences.append(seq)
    return sequences

def undersample_majority_classes(texts, labels, sampling_strategy=0.5):
    label_counts = Counter(labels)
    min_count = min(label_counts.values())
    
    undersampled_texts = []
    undersampled_labels = []
    
    for label in set(labels):
        indices = [i for i, l in enumerate(labels) if l == label]
        if label_counts[label] > min_count:
            target_count = max(min_count, int(label_counts[label] * sampling_strategy))
            selected_indices = random.sample(indices, target_count)
        else:
            selected_indices = indices
        
        undersampled_texts.extend([texts[i] for i in selected_indices])
        undersampled_labels.extend([labels[i] for i in selected_indices])
    
    return undersampled_texts, undersampled_labels

def prepare_data_with_undersampling(file_path: str, max_length: int, batch_size: int, test_size: float = 0.1, val_size: float = 0.1, sampling_strategy: float = 0.5):
    # Load and preprocess data
    texts, labels, classes = load_data(file_path)
    
    # Create vocabulary using all data
    word_to_idx = create_vocabulary(texts, max_size=max_length)
    
    # First split: train+val and test
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    # Second split: train and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, 
        test_size=val_size/(1-test_size), 
        stratify=train_val_labels, 
        random_state=42
    )
    
    # Apply undersampling only to training data
    train_texts_undersampled, train_labels_undersampled = undersample_majority_classes(
        train_texts, train_labels, sampling_strategy
    )
    
    # Create datasets
    train_dataset = TextClassificationDatasetUnder(train_texts_undersampled, train_labels_undersampled, word_to_idx, max_length)
    val_dataset = TextClassificationDatasetUnder(val_texts, val_labels, word_to_idx, max_length)
    test_dataset = TextClassificationDatasetUnder(test_texts, test_labels, word_to_idx, max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, word_to_idx, classes

def load_word2vec(word_to_idx, word2vec_path):
    word2vec = Word2Vec.load(word2vec_path)
    embedding_dim = word2vec.vector_size
    embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))
    for word, i in word_to_idx.items():
        if word in word2vec.wv:
            embedding_matrix[i] = word2vec.wv[word]
        else:
            # Initialize unknown words with random values
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return torch.FloatTensor(embedding_matrix)

def generate_model_name(architecture="LSTM", 
                        embedding_type="Word2Vec", 
                        hidden_dim=82, 
                        n_layers=2, 
                        max_length=50, 
                        balancing_method=None, 
                        version=1):
    folder = "models/"
    file_name = f"{architecture}_{embedding_type}_{hidden_dim}_{n_layers}_{max_length}_{balancing_method}_v{version}"
    while os.path.exists(folder + file_name):
        version += 1
        file_name = f"{architecture}_{embedding_type}_{hidden_dim}_{n_layers}_{max_length}_{balancing_method}_v{version}"
    return file_name

def export_json(name, word_to_index, classes):
    to_export = dict()
    to_export["word_to_idx"] = word_to_index
    to_export["classes"] = classes

    with open(f"meta_info/{name}.json", 'w') as f:
        json.dump(to_export, f)