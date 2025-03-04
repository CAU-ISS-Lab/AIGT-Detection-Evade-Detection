import torch
import torch.nn as nn
import numpy as np

class RLADmodel(nn.Module):
    def __init__(self,tokenizer,model,device,is_aug):
        super(RLADmodel, self).__init__()
        self.tokenizer=tokenizer
        self.model=model
        self.encoder=nn.Sequential(
            nn.Linear(self.model.config.hidden_size,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,2),
            # nn.Dropout(0.3)
        )
        self.detector=nn.Softmax(dim=1)
        self.device=device
        self.is_aug=is_aug

    def forward(self,text,state):
        output=self.tokenizer(text,return_tensors='pt',padding=True,truncation=True)
        if output["input_ids"].size(1) >512:
            output["input_ids"]=output["input_ids"][:, :512]
            output["attention_mask"] = output["attention_mask"][:, :512]

        output=output.to(self.device)
        input_ids = output["input_ids"]
        attention_mask= output["attention_mask"]

        output=self.model(input_ids=input_ids,attention_mask=attention_mask)
        embedding=output.last_hidden_state[:, 0, :]
        feature = self.encoder(embedding)
        if self.is_aug=='gaussian':
            aug_dis=torch.normal(mean=torch.tensor(state[0]), std=torch.tensor(state[1]), size=(self.model.config.hidden_size, ))
        else:
            if self.is_aug=='uniform':
                aug_dis=torch.FloatTensor(self.model.config.hidden_size,).uniform_(state[0],state[1])
            else:
                return feature
        aug_dis=aug_dis.to(self.device)
        aug_embedding=embedding+aug_dis

        aug_feature=self.encoder(aug_embedding)

        return feature,aug_feature


    def predict(self,text):
        # print(text)
        output = self.tokenizer(text,return_tensors='pt',padding=True,truncation=True)
        # print(output)
        if output["input_ids"].size(1) > 512:
            output["input_ids"] = output["input_ids"][:, :512]
            output["attention_mask"] = output["attention_mask"][:, :512]

        output=output.to(self.device)
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]

        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embedding = output.last_hidden_state[:, 0, :]

        feature = self.encoder(embedding)

        res = self.detector(feature)

        return res

class RLADloss(nn.Module):
    def __init__(self,weight_1,weight_2,weight_3):
        super(RLADloss, self).__init__()
        self.ce_loss=nn.CrossEntropyLoss()
        self.detector=nn.Softmax()
        self.weight_1=weight_1
        self.weight_2=weight_2
        self.weight_3 = weight_3

    def forward(self,feature,aug_feature,label):
        loss_1=self.ce_loss(feature,label)
        loss_2=self.ce_loss(aug_feature,label)

        loss_3=torch.norm(feature-aug_feature)**2
        # print(loss_3,loss_2,loss_1)

        total_loss=self.weight_1*loss_1+self.weight_2*loss_2+self.weight_3*loss_3

        prediction=self.detector(feature)

        return total_loss,prediction

class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.detector = nn.Softmax()

    def forward(self,feature,label):
        loss=self.ce_loss(feature,label)
        # print(loss_3,loss_2,loss_1)

        prediction=self.detector(feature)

        return loss,prediction
