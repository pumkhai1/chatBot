import os
print("Creating virtual enviroment________________________")
os.system("python3 -m venv venv") # create env
print("activation virtual enviroment________________________")
os.system(". venv/bin/activate") # activate env

print("installing numpy________________________")
os.system("pip install numpy") # install numpy
print("installing PyTorch________________________")
os.system("pip install torch") # install py torch
print("installing nltk________________________")
os.system("pip install nltk") # install nltk lib
os.system("python download_nltk.py")




