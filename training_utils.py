import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet


def open_json_file(name: str) -> json:
    '''
    :param name: file name
    :return intents: open json file
    '''
    with open(name, 'r') as f:
        intents = json.load (f)
    return intents


def stem_and_lower(all_words: [], tags: [], ignore_words: [] = ['?','.','!' ]) -> []:
    '''
    :param all_words:
    :param tags:
    :param ignore_words:
    :return:
    '''
    # stem and lower each word
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    # remove duplicates and sort
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    return all_words, tags


def create_training_data(tags: [], xy: [], all_words:[], X_train = [], y_train = []) -> np:
    '''
    :param tags:
    :param xy:
    :param all_words:
    :param X_train:
    :param y_train:
    :return:
    '''
    for (pattern_sentence,tag) in xy:
        # X: bag of words for each pattern_sentence
        bag = bag_of_words ( pattern_sentence,all_words )
        X_train.append ( bag )
        # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
        label = tags.index ( tag )
        y_train.append ( label )
    X_train = np.array ( X_train )
    y_train = np.array ( y_train )
    return X_train, y_train


def loss_function(model: NeuralNet, learning_rate: float):
    '''
    :param model:
    :param learning_rate:
    :return:
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam (model.parameters (), lr=learning_rate)
    return criterion, optimizer


def train_model(num_epochs, train_loader, device, model, criterion, optimizer):
    '''
    This train the model
    '''
    for epoch in range(num_epochs):
        for (words,labels) in train_loader:
            words = words.to ( device )
            labels = labels.to ( dtype=torch.long ).to ( device )
            # Forward pass
            outputs = model( words )
            # if y would be one-hot, we must apply
            # labels = torch.max(labels, 1)[1]
            loss = criterion ( outputs,labels )
            # Backward and optimize
            optimizer.zero_grad ()
            loss.backward ()
            optimizer.step ()
        if (epoch + 1) % 100 == 0:
            print ( f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item ():.4f}' )
    print(f'final loss: {loss.item ():.4f}')


def save_data(model: NeuralNet, input_size: int, hidden_size: int, output_size: int, all_words: [], tags: [], FILE: str) -> None:
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }
    torch.save(data, FILE)
    print(f'training complete. file saved to {FILE}')


def appending_ops(intents):
    all_words = [ ]
    tags = [ ]
    xy = [ ]
    # loop through each sentence in our intents patterns
    for intent in intents [ 'intents' ]:
        tag = intent [ 'tag' ]
        # add to tag list
        tags.append ( tag )
        for pattern in intent [ 'patterns' ]:
            # tokenize each word in the sentence
            w = tokenize ( pattern )
            # add to our words list
            all_words.extend ( w )
            # add to xy pair
            xy.append ( (w,tag) )
    return all_words, tags, xy