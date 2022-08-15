from training_utils import open_json_file,stem_and_lower,create_training_data,loss_function,train_model,save_data, \
    appending_ops,create_model_and_device,create_load_dataset

intents = open_json_file('intents.json')
# appending contents from json files
all_words, tags, xy = appending_ops(intents)
# stem and lower each word
ignore_words = ['?', '.', '!']
all_words, tags = stem_and_lower(all_words, tags, ignore_words=ignore_words)
# create training data
X_train, y_train = create_training_data(tags, xy, all_words, X_train=[], y_train=[])
# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
print (input_size, output_size)
# create and load dataset
dataset, train_loader = create_load_dataset(X_train, y_train, batch_size)
# create model and device
device, model = create_model_and_device(input_size, hidden_size, output_size)
# Loss and optimizer
criterion, optimizer = loss_function(model, learning_rate)
# Train the model
train_model(num_epochs, train_loader, device, model, criterion, optimizer)
# save data
FILE = "data.pth"
save_data(model, input_size, hidden_size, output_size, all_words, tags, FILE)