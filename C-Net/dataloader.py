from torch.utils.data import Dataset
import pandas as pd

class CustomTrainDataset(Dataset):
    def __init__(self, machine, text,label):
        self.machine_text = machine
        self.text=text
        self.label = label
        # print(self.label)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        sample = {
            'machine_input': self.machine_text[index],
            'text':self.text[index],
            'label': self.label[index]
        }
        return sample

class CustomTestDataset(Dataset):
    def __init__(self, text,label):
        self.text=text
        self.label = label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        sample = {
            'text':self.text[index],
            'label': self.label[index]
        }
        return sample

def get_train_data(train_data_file):
    train = pd.read_csv(train_data_file)
    train_text = []
    train_labels = []
    machine_text = []
    for _, row_train in train.iterrows():
        train_labels.append(row_train["label"])
        train_text.append(row_train["answer"])
        machine_text.append(row_train['machine_text'])
    train_loader = CustomTrainDataset(machine_text, train_text, train_labels)
    return train_loader

def get_test_data(test_data_file) -> tuple:
    test = pd.read_csv(test_data_file)
    test_text = []
    test_labels = []
    if "TT" not in test_data_file:
        for _, row_test in test.iterrows():
            test_labels.append(row_test["label"])
            test_text.append(row_test["answer"])
        # print(machine_text)
        test_loader=CustomTestDataset(test_text,test_labels)
    else:
        for _, row_test in test.iterrows():
            if row_test["label"]=="human":
                test_labels.append(0)
            else:
                test_labels.append(1)
            test_text.append(row_test["Generation"])
        # print(machine_text)
        test_loader=CustomTestDataset(test_text,test_labels)

    return test_loader