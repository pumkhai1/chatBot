import os

print("Creating and activating virtual env________________________")
os.system("python3 -m venv venv | . venv/bin/activate")
print("Installing numpy________________________")
os.system("pip install numpy") # install numpy
print("Installing PyTorch________________________")
os.system("pip install torch") # install py torch
print("Installing nltk________________________")
os.system("pip install nltk") # install nltk lib
os.system("python download_nltk.py")




