import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

def preprocess_text(text, tokenizer, max_length):
    """
    Preprocess the text data to fit the model's requirements.
    """
    # Tokenize the text
    sequences = tokenizer.texts_to_sequences([text])
    # Pad the sequences
    padded = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded


def predict(text, model, tokenizer, max_length):
    """
    Make a prediction using the pre-trained model.
    """
    # Preprocess the text
    preprocessed_text = preprocess_text(text, tokenizer, max_length)
    # Make a prediction
    prediction = model.predict(preprocessed_text)
    return prediction


def main():

    categories = ['Computer Science', 'Physics', 'Mathematics', 'Statistics', 'Quantitative Biology', 'Quantitative Finance']

    # load the text to classify
    with open('test.txt', 'r') as f:
        # change the name of the file to classify (should be a single line of text
        # containing the title and abstract of the research paper).
        text = f.read()

    # Load the model
    model = load_model('research_paper_classification3.keras', compile=False)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

    tokenizer = Tokenizer(num_words=10000)  # don't change this value
    max_length = 1200  # don't change this value

    # Make a prediction
    prediction = predict(text, model, tokenizer, max_length)
    print("Prediction:", prediction)

    # Show the label corresponding to the predicted label
    print("Category:", categories[prediction.argmax()])
    print("Probability:", prediction.max())



if __name__ == "__main__":
    main()