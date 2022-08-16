# chatBot

I just learn how to implement simple chat-bot from Python-Engineer's youtube channel.
I also create functions that I can reuse them for future NLP chat bot project. I also have python script that can make it easier to deploy and install
necessary package.

# How to Run it on Local Machine
  -  Clone this repo.
  -  run ```python -m venv venv```
  -  run ```. venv/bin/activate``` 
  -  run ``` python install_pkgs.py```
  -  run ``` python run_chatBot.py```

enjoy it and feel free to change the training data in intents.json file.

# Installation files
  - ```install_pkgs.py``` Install PyTorch, Numpy, and NLTK Lib
  - ```run_chatBot.py``` train and run the model and delete the saved data in order to avoid conflict.
  - ```download_nltk.py``` download nltk 
#  Customs files that I modulized codes and function/methods 
  - ```training_utils.py``` necessary methods/function to train model
  - ```nltk_utils.py``` necessary functions/methods for NLP. bag_of_word, tokeniziation, and etc.
