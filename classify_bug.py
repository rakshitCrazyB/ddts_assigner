import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys



stemmer = LancasterStemmer()

# Function to clean data. Lowercase, Tokenize, remove pure numbers

def clean_data(text):
	text = text.lower()
	tokens = nltk.word_tokenize(text)

	newTokens = []
	for unit in tokens:
		if not unit.isdigit():
			newTokens.append(stemmer.stem(unit))

	return newTokens

# a table structure to hold the allowed characters
tbl = dict.fromkeys((i for i in range(256)
	if (not ((i>=48 and i<=57) or (i>=65 and i<=90) or (i>=97 and i<=122)))), ' ')


def remove_punctuation(text):
    return text.translate(tbl)


# initialize the stemmer
stemmer = LancasterStemmer()

data = json.load(open('bugs_samples.json', 'r'))

# A set of all componenents
categories = set()

# A set of all words in training data i.e Global words
words = set()

# Per cdets list with bag of words and component as unit.
docs = []

for each in data:
    categories.add(each['component'])
    each_sentence = remove_punctuation(each['notes'])
    w = clean_data(each_sentence)
    words = words.union(set(w))
    docs.append((w, each['component']))

all_components= sorted(list(categories))

# create our training data
training = []
output = []
output_empty = [0] * len(categories)

for doc in docs:
    bitmap_of_words = []  # 
    # list of tokenized words for the pattern
    token_words = doc[0]

    # create our bag of words array
    for w in words:
        bitmap_of_words.append(1) if w in token_words else bitmap_of_words.append(0)
    # 
    bitmap_of_component = list(output_empty)
    bitmap_of_component[all_components.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells
    # which category that bow belongs to.
    
    training.append([bitmap_of_words, bitmap_of_component])

#training data generated



# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')


# let's test the mdodel for a few sentences:
# the first two sentences are used for training, and the last two sentences are not present in the training data.
sent_1 = "arp resolve issue"
sent_2 = "kernel panic seen"
sent_3 = "VRRP session not established"
sent_4 = "link local address missing"

# a method that takes in a sentence and list of all words
# and returns the data in a form the can be fed to tensorflow


def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1

    return(np.array(bow))


# we can start to predict the results for each of the 4 sentences
print(all_components[np.argmax(model.predict([get_tf_record(sent_1)]))])
print(all_components[np.argmax(model.predict([get_tf_record(sent_2)]))])
print(all_components[np.argmax(model.predict([get_tf_record(sent_3)]))])
print(all_components[np.argmax(model.predict([get_tf_record(sent_4)]))])
