from flask import Flask, render_template, request, jsonify
import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)
app.static_folder = 'static'

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Load model, words, and classes
try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    print("Model loaded successfully!")
except:
    # If model doesn't exist, train it
    print("Model not found. Training new model...")
    
    # Prepare training data
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word in the pattern
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            # Add documents to the corpus
            documents.append((word_list, intent['tag']))
            # Add to classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and lower each word and remove duplicates
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
    words = sorted(list(set(words)))
    classes = sorted(list(set(classes)))

    print(f"Total patterns: {len(documents)}")
    print(f"Tags: {classes}")
    print(f"Unique lemmatized words: {len(words)}")

    # Create training data
    training = []
    output_empty = [0] * len(classes)

    # Create bag of words for each pattern
    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)

        # Create output row with '1' for current tag and '0' for other tags
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])

    # Shuffle training data
    random.shuffle(training)

    # Convert training data to numpy arrays
    train_x = []
    train_y = []
    for bag, output_row in training:
        train_x.append(bag)
        train_y.append(output_row)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # Create and train the model
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(train_x, train_y)

    # Save the model and words
    pickle.dump(model, open('model.pkl', 'wb'))
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    print("Model trained and saved successfully!")

# Function to clean up a sentence
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Function to convert a sentence to a bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Function to predict the class of a sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict_proba([bow])[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Function to get a response
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Maaf, saya tidak mengerti pertanyaan Anda. Bisa tolong diulangi?"
    
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Chatbot function
def chatbot_response(message):
    ints = predict_class(message)
    res = get_response(ints, intents)
    return res

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.json['message']
    response = chatbot_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True, port=5000)