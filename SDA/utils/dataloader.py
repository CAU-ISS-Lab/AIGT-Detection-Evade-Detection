from torch.utils.data import Dataset
import pandas as pd
import os
import json

class prompt_dataset(Dataset):
    def __init__(self,df,source_dict):
        self.df=df
        self.source_dict=source_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        prompt=self.df.iloc[index]['prompt']

        return {
            'text': tokenized_text['input_ids'].flatten(),
            'attention_mask': tokenized_text['attention_mask'].flatten(),
            'label': torch.tensor(label_number)
        }

class dataset(Dataset):
    def __init__(self,data):
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return {
            "prompt":self.data[index]['prompt'],
            "answer":self.data[index]['answer']
        }

def load_data(data_path):
    data=[]
    if '.tsv' in data_path:
        data_df=pd.read_csv(data_path,sep='\t')
        for index,item in data_df.iterrows():
            data.append({"answer":item["SICO-output"],"label":1})
    # elif 'human' in data_path:
    #     with open(data_path,'r',encoding='utf-8') as file:
    #         for line in file:
    #             line_data=json.loads(line)
    #             data.append({"answer":line_data["human"],"label":1})
    # elif 'pp' in data_path:
    #     with open(data_path,'r',encoding='utf-8') as file:
    #         for line in file:
    #             line_data=json.loads(line)
    #             data.append({"answer":line_data["paraphrase_outputs"]["lex_40_order_40"]["output"][0],"label":1})
    else:
        with open(data_path,'r',encoding='utf-8') as file:
            for line in file:
                line_data=json.loads(line)
                data.append(line_data)
    return data