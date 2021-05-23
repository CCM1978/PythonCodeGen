from nltk.stem import WordNetLemmatizer
import pickle
import tensorflow
import json
import numpy as np
import tensorflow.keras
import keras_preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
from pathlib import Path
from tensorflow.keras.models import load_model
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    classes = []
    documents = []
    ignore_letters = ['!', '?', ',', '.']
    with open("intents.json") as file:
        intents = json.load(file)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # tokenize each word
            word = nltk.word_tokenize(pattern)
            words.extend(word)
            # add documents in the corpus
            documents.append((word, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    # lemmaztize and lower each word and remove duplicates
    words = [lemmatizer.lemmatize(w.lower())
             for w in words if w not in ignore_letters]
    words = sorted(list(set(words)))
    # sort classes
    classes = sorted(list(set(classes)))
    # documents = combination between patterns and intents

    # create our training data
    training = []
    # create an empty array for our output
    output_empty = [0] * len(classes)
    # training set, bag of words for each sentence
    for doc in documents:
        # initialize our bag of words
        bag = []
        # list of tokenized words for the pattern
        pattern_words = doc[0]
        # lemmatize each word - create base word, in attempt to represent related words
        pattern_words = [lemmatizer.lemmatize(
            word.lower()) for word in pattern_words]
        # create our bag of words array with 1, if word match found in current pattern
        for word in words:
            bag.append(1) if word in pattern_words else bag.append(0)

        # output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])
    # shuffle our features and turn into np.array
    random.shuffle(training)
    training = np.array(training)
    # create train and test lists. X - patterns, Y - intents
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    pickle.dump((words, classes, train_x, train_y), open('data.pkl', 'wb'))


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

path = Path("chatbot_model.h5")

if path.exists():
    model = load_model('chatbot_model.h5')
else:
    # fitting and saving the model
    hist = model.fit(np.array(train_x), np.array(train_y),
                     epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5', hist)


with open("data.pkl", "rb") as f:
    words, classes, train_x, train_y = pickle.load(f)


def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    bag = [0]*len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)


def string_manipulation(string):
    string = str(string)
    left = "</div></div><div class=\"bot-inbox inbox\"><div class=\"icon\"><i class=\"fas fa-user\"></i></div><div class=\"msg-header\"><p>"
    last = "</p></div></div>"
    string = string.split(" <break> ")
    first = string[0]
    string.pop(0)
    blank = ""
    first = '<p>' + first + '</p>'
    for items in string:
        item = left + items + last
        blank += item
    final = first + blank
    return final


def getResponse(user_input):
    user_input = str(user_input)
    results = model.predict(np.array([bag_of_words(user_input, words)]))[0]
    results_index = np.argmax(results)
    tag = classes[results_index]
    list_of_intents = intents['intents']
    not_found = ["Sorry I didn't get that. Try to rephrase what you just said",
                 "Something asked wrong", "My bad, ask something else"]
    two_messages = ["Uniform"]
    end_tag = "</p></div></div>"
    if results[results_index] > 0.9:
        for i in list_of_intents:
            if (i['tag'] == tag):
                if i['tag'] in two_messages:
                    result = random.choice(i['responses'])
                    result = string_manipulation(result)
                    return result
                else:
                    result = random.choice(i['responses'])
                    result = '<p>' + result + end_tag
                    return result
    else:
        result = '<p>' + random.choice(not_found) + end_tag
        return result


getResponse("dress code")
