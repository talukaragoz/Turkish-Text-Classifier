import os
from llm_query import test_llm
from model_query import TextClassifier

# Initialize the LSTM classifier
lstm_classifier = TextClassifier()

def get_user_input():
    return input("Enter the text to classify (or 'quit' to exit): ")

def main():
    print("Welcome to the Text Classifier!")
    print("This tool will classify your input using both LLM (llama3) and LSTM models.")
    
    while True:
        user_input = get_user_input()
        
        if user_input.lower() == 'quit':
            print("Thank you for using the Text Classifier. Goodbye!")
            break
        
        print("\nClassifying...")
        
        # LLM Classification
        llm_result = test_llm(user_input)
        print(f"LLM Classification: {llm_result}")
        
        # LSTM Classification
        lstm_result = lstm_classifier.classify(user_input)
        print(f"LSTM Classification: {lstm_result}")
        
        print("\n")

if __name__ == "__main__":
    main()