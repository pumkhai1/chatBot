# all imports
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
from ChatDataset import ChatDataset


# all def and functions
def open_file (name):
    with open (name, 'r') as f:
        intents = json.load (f)
    return intents


def append_all_words_tags_and_xy (intents):
    all_words = [ ]
    tags = [ ]
    xy = [ ]

    # loop through each sentence in our intents patterns
    for intent in intents [ 'intents' ]:
        tag = intent [ 'tag' ]
        # add to tag list
        tags.append (tag)
        for pattern in intent [ 'patterns' ]:
            # tokenize each word in the sentence
            w = tokenize (pattern)
            # add to our words list
            all_words.extend (w)
            # add to xy pair
            xy.append ((w, tag))
    return all_words, tags, xy


def stem_and_lower (all_words, tags, ignore_words=[ '?', '.', '!' ]):
    ignore_words = [ '?', '.', '!' ]
    all_words = [ stem (w) for w in all_words if w not in ignore_words ]
    # remove duplicates and sort
    all_words = sorted (set (all_words))
    tags = sorted (set (tags))
    return all_words, tags


def create_training_data (xy, all_words, ):
    X_train = [ ]
    y_train = [ ]
    for (pattern_sentence, tag) in xy:
        # X: bag of words for each pattern_sentence
        bag = bag_of_words (pattern_sentence, all_words)
        X_train.append (bag)
        # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
        label = tags.index (tag)
        y_train.append (label)
    X_train = np.array (X_train)
    y_train = np.array (y_train)
    return X_train, y_train


def hyper_param (num_epochs=1000,
                 batch_size=8,
                 learning_rate=0.001,
                 input_size=0,
                 hidden_size=8,
                 output_size=0,
                 ):
    num_epochs = num_epochs
    batch_size = batch_size
    learning_rate = learning_rate
    input_size = input_size  # len (X_train [ 0 ])
    hidden_size = hidden_size
    output_size = output_size  # len (tags)
    print (input_size, output_size)
    return num_epochs, batch_size, learning_rate, input_size, hidden_size, output_size


def create_dataset_device_and_model(batch_size, input_size, hidden_size):
    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=0)
    device = torch.device ('cuda' if torch.cuda.is_available () else 'cpu')
    model = NeuralNet (input_size, hidden_size, hidden_size).to (device)

    return train_loader, device, model


def loss_and_optimizer(model, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters (), lr=learning_rate)
    return criterion, optimizer


# all other code
intents = open_file ('intents.json')
# loop through each sentence in our intents patterns
all_words, tags, xy = append_all_words_tags_and_xy (intents)
# stem and lower each word
ignore_words = [ '?', '.', '!' ]
all_words, tags = stem_and_lower (all_words, tags, ignore_words=ignore_words)

# create training data
X_train, y_train = create_training_data (xy, all_words)
# Hyper-parameters
num_epochs, batch_size, learning_rate, input_size, hidden_size, output_size = hyper_param (num_epochs=1000,
                                                                                           batch_size=8,
                                                                                           learning_rate=0.001,
                                                                                           input_size=len(
                                                                                               X_train [0]),
                                                                                           hidden_size=8,
                                                                                           output_size=len(tags),
                                                                                           )
#create dataset, device and model
train_loader, device, model = create_dataset_device_and_model(batch_size, input_size, hidden_size)

# Loss and optimizer
criterion, optimizer = loss_and_optimizer(model, learning_rate)

# Train the model
for epoch in range (num_epochs):
    for (words, labels) in train_loader:
        words = words.to (device)
        labels = labels.to (dtype=torch.long).to (device)

        # Forward pass
        outputs = model (words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion (outputs, labels)

        # Backward and optimize
        optimizer.zero_grad ()
        loss.backward ()
        optimizer.step ()

    if (epoch + 1) % 100 == 0:
        print (f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item ():.4f}')

print (f'final loss: {loss.item ():.4f}')

data = {
    "model_state": model.state_dict (),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save (data, FILE)
print (f'training complete. file saved to {FILE}')
