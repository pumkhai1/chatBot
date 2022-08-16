import random
import torch
from model import NeuralNet
from nltk_utils import bag_of_words,tokenize
from training_utils import open_json_file
import os.path


# create bot OOP style

class CreateBot:
    def __init__ (self,name):
        self.__bot_name = name
        if not os.path.isfile ( 'data.pth' ):
            self.__train_model ()

    def __get_name__ (self):
        return self.__bot_name

    def __set_name__ (self,name):
        self.__bot_name = name

    def __retrained__ (self):
        return self.__train_model ()

    def __response (self):
        print ( f"Hi! I am {self.__bot_name}. A virtual assistant. (type 'quit' to exit)" )
        while True:
            sentence = input ( "You: " )
            if sentence == "quit":
                break
            sentence = tokenize ( sentence )
            X = bag_of_words ( sentence,self.__all_words )
            X = X.reshape ( 1,X.shape [ 0 ] )
            X = torch.from_numpy ( X ).to ( self.__device )

            output = self.__model ( X )
            _,predicted = torch.max ( output,dim=1 )

            tag = self.__tags [ predicted.item () ]

            probs = torch.softmax ( output,dim=1 )
            prob = probs [ 0 ] [ predicted.item () ]
            if prob.item () > 0.75:
                for intent in self.__intents [ 'intents' ]:
                    if tag == intent [ "tag" ]:
                        print ( f"{self.__bot_name}: {random.choice ( intent [ 'responses' ] )}" )
            else:
                print ( f"{self.__bot_name}: Sorry, I do not understand..." )
        print ( "Have a nice day." )

    def run (self):
        self.__load_data ()
        self.__eval_model ()
        self.__response ()

    def __load_data (self):
        self.__intents = open_json_file ( 'intents.json' )
        self.__FILE = "data.pth"
        self.__data = torch.load ( self.__FILE )
        self.__input_size = self.__data [ "input_size" ]
        self.__hidden_size = self.__data [ "hidden_size" ]
        self.__output_size = self.__data [ "output_size" ]
        self.__all_words = self.__data [ 'all_words' ]
        self.__tags = self.__data [ 'tags' ]
        self.__model_state = self.__data [ "model_state" ]

    def __eval_model (self):
        self.__device = torch.device ( 'cuda' if torch.cuda.is_available () else 'cpu' )
        self.__model = NeuralNet ( self.__input_size,self.__hidden_size,self.__output_size ).to ( self.__device )
        self.__model.load_state_dict ( self.__model_state )
        self.__model.eval ()

    def __train_model (self):
        os.system ( 'python train.py' )
