import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Prepare data
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

# Save words and classes to pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

print("words.pkl and classes.pkl created successfully!")