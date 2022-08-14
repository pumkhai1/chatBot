import os
# train it first then run
os.system("python train.py")
os.system("python chat.py")
os.system("rm -rf data.pth")
