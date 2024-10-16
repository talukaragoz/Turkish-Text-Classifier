# Turkish Text Classification Project

This project implements a Turkish text classification system using both a Long Short-Term Memory (LSTM) neural network and a Large Language Model (LLM) approach. It provides an interactive command-line interface for classifying Turkish text into various domains.

## Getting Started

### Prerequisites

- Docker
- Ollama with the `llama3` model installed locally

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/talukaragoz/turkish-text-classification.git
   cd turkish-text-classification
   ```

2. Ensure you have the `llama3` model downloaded locally for Ollama:
   ```
   ollama pull llama3
   ```

## Running the Application

### Docker Deployment (Recommended)

1. Build the Docker image:
   ```
   docker build -t interactive-text-classifier .
   ```

2. Run the Docker container:
   ```
   docker run -it -v ~/.ollama/models:/root/.ollama/models:ro interactive-text-classifier
   ```

   This command mounts your local Ollama models directory to the container, allowing it to use your locally installed `llama3` model.

3. The interactive classifier will start, prompting you to enter text for classification.

### Local Development (Alternative)

If you prefer to run the application without Docker:

1. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

2. Run the interactive classifier:
   ```
   python interactive_classify.py
   ```

## Using the Interactive Classifier

Once the application is running, you can:

1. Enter Turkish text when prompted.
2. The system will classify the text using both the LSTM and LLM models.
3. Results from both models will be displayed.
4. Type 'quit' to exit the application.

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

## Additional Deployment Option: FastAPI Service

For those who need a web service instead of an interactive command-line interface, this project also includes a FastAPI-based deployment option.

### Running with Docker Compose

1. Ensure you have Docker and Docker Compose installed.
2. Run the following command in the project directory:
   ```
   docker-compose up --build
   ```
3. The API will be available at `http://localhost:8000`.

### API Endpoints

- `/classify_llm`: Classify text using the LLM approach
- `/classify_lstm`: Classify text using the LSTM model
- `/classify_both`: Classify text using both LLM and LSTM approaches
- `/batch_classify_lstm`: Batch classify multiple texts using the LSTM model

## Project Structure

- `interactive_classify.py`: Main script for interactive classification
- `app.py`: FastAPI application and API endpoints (for web service option)
- `model.py`: LSTM model architecture
- `model_query.py`: LSTM model inference
- `llm_query.py`: LLM-based classification
- `training.py`: LSTM model training script
- `utils.py`: Utility functions for data processing and model helpers
- `requirements.txt`: Python dependencies
- `Dockerfile`: Docker configuration file
- `docker-compose.yml`: Docker Compose configuration (for web service option)