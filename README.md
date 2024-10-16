# Turkish Text Classification Project

This project implements a Turkish text classification system using both a Long Short-Term Memory (LSTM) neural network and a Large Language Model (LLM) approach. It provides a FastAPI web service for classifying Turkish text into various domains.

## Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose (optional, for containerized deployment)
- Ollama (for LLM-based classification)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/turkish-text-classification.git
   cd turkish-text-classification
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the `llama3` model downloaded locally for Ollama. Even when running in Docker, the model should be stored on your local machine:
   ```
   ollama pull llama3
   ```

## Running the Application

### Local Development

1. Start the FastAPI server:
   ```
   uvicorn app:app --reload
   ```

2. The API will be available at `http://localhost:8000`.

### Docker Deployment

1. Build and start the Docker containers:
   ```
   docker-compose up --build
   ```

2. The API will be available at `http://localhost:8000`.

## API Endpoints

- `/classify_llm`: Classify text using the LLM approach
- `/classify_lstm`: Classify text using the LSTM model
- `/classify_both`: Classify text using both LLM and LSTM approaches
- `/batch_classify_lstm`: Batch classify multiple texts using the LSTM model

## Models

### LSTM Classifier

The LSTM classifier is implemented in `model.py` and uses a bidirectional LSTM architecture. It's trained on a dataset of Turkish text excerpts and their corresponding domain labels.

Key features:
- Embedding layer initialized with Word2Vec embeddings
- Bidirectional LSTM layers
- Dropout for regularization
- Linear layer for classification

### LLM Classifier

The LLM classifier uses the `llama3` model through Ollama. It's implemented in `llm_query.py` and uses a prompt-based approach to classify text into predefined domains.

## Training Custom LSTM Classifiers

You can train your own LSTM classifiers using the `training.py` script. This script handles data preprocessing, model training, and evaluation.

To train a new model:

1. Prepare your dataset in the format specified in the `load_data` function in `utils.py`.
2. Adjust hyperparameters in `training.py` if needed.
3. Run the training script:
   ```
   python training.py
   ```

The script will:
- Load and preprocess the data
- Create a vocabulary
- Initialize the model with pre-trained Word2Vec embeddings
- Train the model using the specified hyperparameters
- Evaluate the model on a test set
- Save the trained model and associated metadata

## Project Structure

- `app.py`: FastAPI application and API endpoints
- `model.py`: LSTM model architecture
- `model_query.py`: LSTM model inference
- `llm_query.py`: LLM-based classification
- `training.py`: LSTM model training script
- `utils.py`: Utility functions for data processing and model helpers
- `requirements.txt`: Python dependencies
- `Dockerfile` and `docker-compose.yml`: Docker configuration files