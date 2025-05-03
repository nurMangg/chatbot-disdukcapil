import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents
with open('intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Check if words.pkl and classes.pkl already exist
try:
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    print("Menggunakan words.pkl dan classes.pkl yang sudah ada")
except:
    print("Membuat file words.pkl dan classes.pkl yang baru...")
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

    print(f"Total pattern: {len(documents)}")
    print(f"Tag: {classes}")
    print(f"Jumlah kata yang dilemmatkan: {len(words)}")

    # Save words and classes to pickle files
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))
    print("words.pkl dan classes.pkl berhasil dibuat!")

# Create training data
print("Membuat data pelatihan...")
training = []
output_empty = [0] * len(classes)

# Create bag of words for each pattern
documents = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern
        word_list = nltk.word_tokenize(pattern)
        # Lemmatize words
        word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list]
        # Add to documents list
        documents.append((word_list, intent['tag']))

# Create training data with bag of words
for document in documents:
    bag = []
    word_patterns = document[0]
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
train_y = np.argmax(train_y, axis=1)


print(f"Data pelatihan shape: X={train_x.shape}, Y={train_y.shape}")

# Create and train the model
print("Melatih model...")
model = MultinomialNB()
model.fit(train_x, train_y)

# Save the model
pickle.dump(model, open('model.pkl', 'wb'))
print("Model berhasil dilatih dan disimpan di model.pkl!")

# Test the model with a few examples
def bersihkan_kalimat(kalimat):
    words = nltk.word_tokenize(kalimat)
    words = [lemmatizer.lemmatize(word.lower()) for word in words]
    return words

def bag_of_words(kalimat):
    words = bersihkan_kalimat(kalimat)
    bag = [0] * len(words)
    for w in words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def prediksi_kelas(kalimat):
    bow = bag_of_words(kalimat)
    res = model.predict_proba([bow])[0]
    ERROR_THRESHOLD = 0.25
    hasil = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # Sort by probability
    hasil.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in hasil:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# Test with a few examples
kalimat_uji = [
    "Halo, selamat pagi",
    "Bagaimana cara membuat KTP?",
    "Jam berapa kantor Disdukcapil buka?",
    "Terima kasih atas informasinya"
]

print("\nMenguji model dengan beberapa contoh:")
for kalimat in kalimat_uji:
    ints = prediksi_kelas(kalimat)
    print(f"\nInput: {kalimat}")
    if ints:
        print(f"Prediksi intent: {ints[0]['intent']} (probability: {ints[0]['probability']})")
    else:
        print("Tidak ada intent yang terdeteksi di atas ambang")
