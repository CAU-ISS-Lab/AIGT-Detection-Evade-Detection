import faiss
import numpy as np
import pandas as pd
from transformers import AutoTokenizer,AutoModel
import os
from tqdm import tqdm
import pickle
import torch
from utils.dataloader import load_data

class FaissRAG:
    def __init__(self,device,data_path,index_path,model_path='model/roberta-base'):
        self.tokenizer=AutoTokenizer.from_pretrained(model_path)
        self.model=AutoModel.from_pretrained(model_path)
        self.index_path=index_path
        self.data_path=data_path
        self.index=None
        self.device=device
        self.model.to(self.device)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if os.path.exists(self.index_path):
            self.load_data()
        else:
            self.indices,self.data=self.index_construct()
    
    def index_construct(self):
        # with open(self.data_path,'r') as file:
        indices=None
        data=load_data(self.data_path)
        for row in tqdm(data):
            text=row['prompt'].replace('\n',' ')
            tokenized_text=self.tokenizer(text,padding='max_length',truncation=True,return_tensors='pt',max_length=512).to(self.device)
            with torch.no_grad():
                encode=self.model(**tokenized_text)
            embedding=encode[0].mean(dim=1).detach().cpu().squeeze().numpy()
            dimension=embedding.shape[0]
            if not indices:
                indices=faiss.IndexFlatL2(dimension)
            indices.add(np.expand_dims(embedding, axis=0))
        self.save_data(indices)
        return indices,data

    def save_data(self,indices):
        faiss.write_index(indices,self.index_path)
        print("indices saved at {}".format(self.index_path))
    
    def load_data(self):
        self.data=load_data(self.data_path)
        self.indices=faiss.read_index(self.index_path)
        print("dataset loaded!")
    
    def search(self,query,example_size):
        tokenized_text=self.tokenizer(query,padding='max_length',truncation=True,return_tensors='pt',max_length=512).to(self.device)
        encode=self.model(**tokenized_text)[0].mean(dim=1).detach().cpu().squeeze().numpy()
        distances,idx=self.indices.search(np.expand_dims(encode,axis=0),example_size)
        result=[self.data[i] for i in idx[0]]
        return result
