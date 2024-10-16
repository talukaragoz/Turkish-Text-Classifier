import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm
from gensim.models import Word2Vec

from utils import *
from model import *

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            
            predictions = model(texts)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(predictions, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Compute AUC-ROC for binary classification
    if len(set(all_labels)) == 2:
        auc_roc = roc_auc_score(all_labels, all_preds)
    else:
        auc_roc = None
    
    return avg_loss, accuracy, f1, auc_roc

def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=5, device='cuda'):
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(texts)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluate on validation set
        val_loss, val_accuracy, val_f1, val_auc_roc = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f}, Accuracy: {val_accuracy:.3f}, F1: {val_f1:.3f}')
        if val_auc_roc is not None:
            print(f'\t Val. AUC-ROC: {val_auc_roc:.3f}')
    
    return model

if __name__ == "__main__":
    file_path = "TWNERTC_All_Versions/TWNERTC_TC_Fine Grained NER_DomainIndependent_NoiseReduction.DUMP"
    max_length = 50
    batch_size = 64
    
    train_loader, val_loader, test_loader, word_to_idx, classes = prepare_data_with_undersampling(
        file_path, max_length, batch_size, sampling_strategy=0.5
    )
    
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    print(f"Number of classes: {len(classes)}")
        
    vocab_size = len(word_to_idx)
    pad_idx = word_to_idx['<pad>']
    
    word2vec_path = "Word2Vec/Word2Vec_large.model"

    # Load Word2Vec model and get embedding dimension
    word2vec = Word2Vec.load(word2vec_path)
    embed_dim = word2vec.vector_size
    
    hidden_dim = 82
    output_dim = len(classes)  # number of classes
    n_layers = 2
    bidirectional = True
    dropout = 0.22
    learning_rate = 0.001
    epochs = 15
    
    model = LSTMClassifier(vocab_size, 
                           embed_dim, 
                           hidden_dim, 
                           output_dim, 
                           n_layers, 
                           bidirectional, 
                           dropout, 
                           pad_idx)
    
    # Load pre-trained embeddings
    pretrained_embeddings = load_word2vec(word_to_idx, word2vec_path)
    model.embedding.weight.data.copy_(pretrained_embeddings)
    model.embedding.weight.requires_grad = True  # False to freeze embeddings
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    device = torch.device('mps' if torch.mps.is_available() else 'cpu') # Choose your desired version
    
    print(f"Training on: {device}")
    
    model = train_model(model, train_loader, val_loader, optimizer, criterion, epochs=epochs, device=device)

    # Evaluate on test set
    test_loss, test_accuracy, test_f1, test_auc_roc = evaluate_model(model, test_loader, criterion, device)
    print("Test Set Performance:")
    print(f"Loss: {test_loss:.3f}, Accuracy: {test_accuracy:.3f}, F1: {test_f1:.3f}")
    if test_auc_roc is not None:
        print(f"AUC-ROC: {test_auc_roc:.3f}")
    
    # Save model and util files
    model_name = generate_model_name("LSTM",
                                     "Word2Vec",
                                     hidden_dim=hidden_dim,
                                     n_layers=n_layers,
                                     max_length=max_length,
                                     balancing_method="undersample")
    
    torch.save(model, f"models/{model_name}.pt")
    export_json(model_name, word_to_idx, classes.tolist())