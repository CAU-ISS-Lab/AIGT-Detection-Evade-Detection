import torch
import transformers
import torch.nn as nn
from transformers import RobertaModel,RobertaTokenizer
from npc_gzip.compressors.gzip_compressor import GZipCompressor
from npc_gzip.compressors.bz2_compressor import Bz2Compressor
from npc_gzip.compressors.lzma_compressor import LzmaCompressor
from npc_gzip.exceptions import (
    CompressedValuesEqualZero,
    InputLabelEqualLengthException,
    InvalidObjectTypeException,
    UnsupportedDistanceMetricException,
)
import math
from sklearn.manifold import TSNE

def cal_ncd(
        compressed_value_a: float,
        compressed_value_b: float,
        compressed_value_ab: float,
) -> float:
    denominator = max(compressed_value_a, compressed_value_b)
    if denominator == 0:
        raise CompressedValuesEqualZero(
            compressed_value_a, compressed_value_b, function_name="Distance._ncd"
        )

    numerator = compressed_value_ab - min(compressed_value_a, compressed_value_b)
    distance = numerator / denominator
    return distance

class CNetModel(nn.Module):
    def __init__(self, model,tokenizer):
        super(CNetModel, self).__init__()
        self.tokenizer=tokenizer
        self.roberta= model
        hidden_size = self.roberta.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
        self.softmax= nn.Softmax(dim=1)
        self.compressor = GZipCompressor()
        self.tokenizer.pad_token = tokenizer.eos_token

    def forward(self, machine_text, text,device):
        ncd_distance=[]
        for index in range(len(machine_text)):
            compressed_machine=self.compressor.get_compressed_length(machine_text[index])
            compressed_text=self.compressor.get_compressed_length(text[index])
            combined = f"{machine_text[index]} {text[index]}"
            compressed_combined=self.compressor.get_compressed_length(combined)
            ncd_distance.append(cal_ncd(compressed_machine,compressed_text,compressed_combined))



        output1=self.tokenizer(machine_text,return_tensors='pt',padding=True,truncation=True)
        output2=self.tokenizer(text,return_tensors='pt',padding=True,truncation=True)

        if output1["input_ids"].size(1) >512:
            output1["input_ids"]=output1["input_ids"][:, :512]
            output1["attention_mask"] = output1["attention_mask"][:, :512]

        if output2["input_ids"].size(1) >512:
            output2["input_ids"]=output2["input_ids"][:, :512]
            output2["attention_mask"] = output2["attention_mask"][:, :512]

        output1=output1.to(device)
        output2 = output2.to(device)

        input_ids_1 = output1["input_ids"]
        attention_mask_1 = output1["attention_mask"]

        input_ids_2 = output2["input_ids"]
        attention_mask_2 = output2["attention_mask"]

        output1 = self.roberta(input_ids=input_ids_1, attention_mask=attention_mask_1)
        output2 = self.roberta(input_ids=input_ids_2, attention_mask=attention_mask_2)
        embedding1 = output1.last_hidden_state[:, 0, :]
        embedding2 = output2.last_hidden_state[:, 0, :]

        similarity = torch.cosine_similarity(embedding1, embedding2, dim=1)

        output = self.classifier(embedding2)

        return output, similarity, ncd_distance

    def predict(self,input,device):
        output = self.tokenizer(input, return_tensors='pt', padding=True, truncation=True,)
        output=output.to(device)
        if output["input_ids"].size(1) >512:
            output["input_ids"]=output["input_ids"][:, :512]
            output["attention_mask"] = output["attention_mask"][:, :512]
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]

        output1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        embedding = output1.last_hidden_state[:, 0, :]

        output = self.classifier(embedding)
        output = self.softmax(output)

        return output,embedding

class CNetLoss(nn.Module):
    def __init__(self):
        super(CNetLoss, self).__init__()
        self.softmax=nn.Softmax(dim=1)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, output,target,similarity, ncd_distance,weight_1,weight_2,weight_3,device):
        loss1 = self.cross_entropy_loss(output, target)

        output=self.softmax(output)
        # similarity loss
        similarity_loss =torch.mean( torch.abs(output[:,1]-torch.exp(similarity-1)))

        # ncd loss
        ncd_distance_tensor = torch.tensor(ncd_distance, dtype=torch.float32)
        ncd_distance_tensor=ncd_distance_tensor.to(device)
        ncd_loss =torch.mean( torch.abs(output[:,1]-torch.exp(-ncd_distance_tensor)))


        total_loss = weight_1*loss1 + weight_2*similarity_loss + weight_3*ncd_loss
        return total_loss,output

