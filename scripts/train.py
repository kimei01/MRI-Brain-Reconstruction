import json

from preprocessing.data_processing import data_processing
from preprocessing.data_representation import create_representations
from scripts.data_loader import create_dataloaders

#flair, t1, t2..etc 
sequence = ""

#Load the splits
with open(f"splits/{sequence}.json", "r") as f:
    split = json.load(f)

#Preprocess each split
x_train, y_train = data_processing(split["train"])
x_val, y_val = data_processing(split["val"])
x_test, y_test = data_processing(split["test"])

#input representations
x_train_complex, x_train_ri = create_representations(x_train)
x_val_complex, x_val_ri = create_representations(x_val)
x_test_complex, x_test_ri = create_representations(x_test)

#Create DataLoaders
train_data = (x_train_complex, x_train_ri, y_train)
val_data = (x_val_complex, x_val_ri, y_val)
test_data = (x_test_complex, x_test_ri, y_test)

train_loader, val_loader, test_loader = create_dataloaders(
    train_data,
    val_data,
    test_data,
    batch_size=2
)
