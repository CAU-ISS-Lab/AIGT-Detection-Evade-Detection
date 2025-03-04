import os.path

from torch.utils.data import Dataset
import pandas as pd
import json
from torchvision import transforms
class DPnetDataset(Dataset):
    def __init__(self, text,label):
        self.text=text
        self.label = label  # 假设这是目标数据

    def __len__(self):
        return len(self.text)  # 返回数据集的样本数量

    def __getitem__(self, index):
        # 根据给定的索引 index 返回对应的数据样本
        sample = {
            'text':self.text[index],
            'label': self.label[index]  # 目标数据
        }
        return sample


def get_data(data_file) -> tuple:
    text = []
    labels = []
    with open(data_file, 'r',encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)  # 解析JSON行
            if 'peerread' in data_file:
                for i in range(len(data['machine_text'])):
                    text.append(data['machine_text'][i])
                    labels.append(1)
                for j in range(len(data['human_text'])):
                    text.append(data['human_text'][j])
                    labels.append(0)
            else:
                if 'paug' in data_file:
                    text.append(data['machine_text_revised'])
                    labels.append(1)
                    text.append(data['human_text'])
                    labels.append(0)
                elif 'raid' in data_file:
                    text.append(data['generation'])
                    if data['model']=='human':
                        labels.append(0)
                    else:
                        labels.append(1)
                elif 'dipper' in data_file:
                    text.append(data['paraphrase_outputs']['lex_40_order_40']['final_input'][0])
                    labels.append(1)
                    text.append(data['human_text'])
                    labels.append(0)
                else:
                    text.append(data['machine_text'])
                    labels.append(1)
                    text.append(data['human_text'])
                    labels.append(0)

        loader = DPnetDataset(text, labels)

    return loader
