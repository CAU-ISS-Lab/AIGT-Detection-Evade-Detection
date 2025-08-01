import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForCausalLM, GPT2Model,GPT2Tokenizer,PreTrainedModel,  Trainer, AutoModel,AutoConfig,RobertaModel,RobertaTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset
import numpy as np
import torch
import time
import glob
import os
import json
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import List, Optional, Tuple, Union
from fuzzywuzzy import fuzz
from sklearn.linear_model import LogisticRegression
from joblib import dump, load
from prompt_constructor import llm_api
from utils.dataloader import load_data
import time

class test_dataset(Dataset):
    def __init__(self,data):
        self.data=data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        if "label" in self.data[index]:
            answer=self.data[index]['label']
        else:
            answer=0
            # if self.data[index]['model']=='human':
            #     answer=0
            # else:
            #     answer=1
        return {
            # "text":self.data[index]["paraphrase_outputs"]["lex_40_order_40"]["output"][0],
            "text":self.data[index]['answer'],
            # "text":self.data[index]['generation'],
            "label":answer
            # "label":1
        }

class RobertaDetector():
    def __init__(self,device,is_mpu=False):
        if is_mpu:
            self.tokenizer = AutoTokenizer.from_pretrained("../model/MPU")
            self.model = AutoModelForSequenceClassification.from_pretrained("../model/MPU")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("../evade/chatgpt-detector")
            self.model = AutoModelForSequenceClassification.from_pretrained("../evade/chatgpt-detector")
        self.device=device
        self.model.to(self.device)
    
    def __call__(self,text):
        with torch.no_grad():
            inputs=self.tokenizer(text,return_tensors='pt', padding=True, truncation=True).to(self.device)
            pred=self.model(**inputs)
            logits=pred.logits
            # print(logits)
            prob = torch.nn.functional.softmax(logits, dim=-1)
            # print(prob)
            pred_class=logits.argmax().item()
            # print(pred_class)
        return pred_class,prob[0]
    
    def evaluate(self,dataloader):
        self.model.eval()
        with tqdm(dataloader) as loop:
            all_targets=[]
            all_pred=[]
            st=time.time()
            for data in loop:
                text=data['text']
                label=data['label']
                with torch.no_grad():
                    inputs=self.tokenizer(text,return_tensors='pt', padding=True, truncation=True,max_length=512).to(self.device)
                    pred=self.model(**inputs)
                    logits=pred.logits
                    pred_class=logits.argmax(dim=-1)
                all_targets.extend(np.array(label))
                all_pred.extend(pred_class.detach().cpu().numpy())
                batch_accuracy=accuracy_score(all_targets,all_pred)
                loop.set_postfix({'acc':batch_accuracy})
            et=time.time()
        print("evaluation time:",et-st)
        accuracy = accuracy_score(all_targets, all_pred)
        auroc = roc_auc_score(all_targets, all_pred)
        precision = precision_score(all_targets, all_pred)
        recall = recall_score(all_targets, all_pred)
        # print(all_targets,all_pred)
        f1 = f1_score(all_targets, all_pred)
        res_dict={'accuracy':{accuracy},'auroc':{auroc},'precision':{precision},'recall':{recall},'f1':{f1}}
        # res_dict={'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1}
        return res_dict

class ProbEstimator:
    def __init__(self, ref_path):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')


    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)

class FastdetectGPT():
    def __init__(self,scoring_model_name,reference_model_name,ref_path='./local_infer_ref',dataset="xsum",device='cuda:1',cache_dir='./chahe'):
        self.scoring_tokenizer = self.load_tokenizer(scoring_model_name, dataset, cache_dir)
        self.scoring_model = self.load_model(scoring_model_name, device, cache_dir)
        self.scoring_model.eval()
        self.scoring_model_name=scoring_model_name
        self.reference_model_name=reference_model_name
        self.ref_path=ref_path
        self.device=device
        if reference_model_name != scoring_model_name:
            self.reference_tokenizer = self.load_tokenizer(reference_model_name, dataset, cache_dir)
            self.reference_model = self.load_model(reference_model_name, device, cache_dir)
            self.reference_model.eval()
        self.criterion_fn = self.get_sampling_discrepancy_analytic
        self.prob_estimator = ProbEstimator(ref_path)

    def __call__(self,text):
        tokenized = self.scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = self.soring_modecl(**tokenized).logits[:, :-1]
            if self.reference_model_name == self.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = self.reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = self.reference_model(**tokenized).logits[:, :-1]
            crit = self.criterion_fn(logits_ref, logits_score, labels)
            # estimate the probability of machine generated text
        prob = self.prob_estimator.crit_to_prob(crit)
        pred_class = 1 if prob > 0.5 else 0
        return pred_class,prob

    
    def load_model(self,model_name, device, cache_dir):
        print(f'Loading model {model_name}...')
        model_kwargs = {}
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f'Moving model to {device}', end='', flush=True)
        start = time.time()
        model.to(device)
        print(f'DONE ({time.time() - start:.2f}s)')
        return model

    def load_tokenizer(self,model_name, for_dataset, cache_dir):
        optional_tok_kwargs = {}
        optional_tok_kwargs['padding_side'] = 'right'
        base_tokenizer = self.from_pretrained(GPT2Tokenizer, model_name, optional_tok_kwargs, cache_dir=cache_dir)
        if base_tokenizer.pad_token_id is None:
            base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
            # if '13b' in model_fullname:
            #     base_tokenizer.pad_token_id = 0
        return base_tokenizer
    
    def get_sampling_discrepancy_analytic(self,logits_ref, logits_score, labels):
        assert logits_ref.shape[0] == 1
        assert logits_score.shape[0] == 1
        assert labels.shape[0] == 1
        if logits_ref.size(-1) != logits_score.size(-1):
            # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
        discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        discrepancy = discrepancy.mean()
        return discrepancy.item()

    def evaluate(self,loader):
        all_labels=[]
        all_preds=[]
        with tqdm(loader) as loop:
            for data in loop:
                text=data["text"]
                label=data['label']
            # evaluate text
                tokenized = self.scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False,truncation=True,max_length=512).to(self.device)
                labels = tokenized.input_ids[:, 1:]
                with torch.no_grad():
                    logits_score = self.scoring_model(**tokenized).logits[:, :-1]
                    if self.reference_model_name == self.scoring_model_name:
                        logits_ref = logits_score
                    else:
                        tokenized = self.reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False,truncation=True,max_length=512).to(self.device)
                        assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                        logits_ref = self.reference_model(**tokenized).logits[:, :-1]
                    crit = self.criterion_fn(logits_ref, logits_score, labels)
            # estimate the probability of machine generated text
                prob = self.prob_estimator.crit_to_prob(crit)
                pred_class = 1 if prob > 0.5 else 0
                    # print(label)
                all_labels.extend(label.tolist())
                all_preds.append(pred_class)
                batch_accuracy = accuracy_score(all_labels, all_preds)
                loop.set_postfix({'acc': batch_accuracy})
        acc=accuracy_score(all_labels,all_preds)
        f1=f1_score(all_labels,all_preds)
        print(f"results: acc: {acc}, f1: {f1}")
        return {"accuracy":acc,"f1 score":f1}

    def from_pretrained(self,cls, model_name, kwargs, cache_dir):
    # use local model if it exists
        local_path = os.path.join(cache_dir, 'local.' + model_name.replace("/", "_"))
        if os.path.exists(local_path):
            return cls.from_pretrained(local_path, **kwargs)
        return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir)

class RADAR():
    def __init__(self,device='cuda:1'):
        self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained("../model/RADAR")
        self.model=AutoModelForSequenceClassification.from_pretrained("../model/RADAR")
        self.model.eval()
        self.model.to(self.device)
    
    def evaluate(self,loader):
        all_predictions=[]
        all_targets=[]
        all_outputs=[]
        machine_prob=[]
        human_prob=[]
        with tqdm(loader) as loop:
            for sample_batch in loop:
                input=sample_batch['text']
                targets=sample_batch['label']
                input = self.tokenizer(input, padding=True, truncation=True, max_length=512, return_tensors="pt")
                input = {k: v.to(self.device) for k, v in input.items()}
                outputs = F.log_softmax(self.model(**input).logits, -1)[:, 0].exp().tolist()
                predictions = [0 if outputs[i]<0.5 else 1 for i in
                        range(len(outputs))]  # Change this based on your output format
                machine_prob.append(outputs[0])
                human_prob.append(1-outputs[0])
        # print(outputs[0],predictions)

            # Collect predictions and targets for metrics calculation
                all_predictions.extend(np.array(predictions))
                all_targets.extend(targets.cpu().numpy())
                all_outputs.append(outputs)

            # 计算当前批次的准确率
                batch_accuracy = accuracy_score(all_targets, all_predictions)
                loop.set_postfix(acc=batch_accuracy)
        accuracy = accuracy_score(all_targets, all_predictions)
        # auroc = roc_auc_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        print(
            f" Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        return {"accuracy":accuracy,"f1 score":f1}

class NPnet(nn.Module):
    def __init__(self,device,is_aug):
        super(NPnet, self).__init__()
        self.tokenizer= RobertaTokenizer.from_pretrained("model/roberta-base")
        self.model= RobertaModel.from_pretrained("model/roberta-base")
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
        self.model.to(self.device)
        self.model.requires_grad_=False

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
        output = self.tokenizer(text,return_tensors='pt',padding=True,truncation=True)
        if output["input_ids"].size(1) > 512:
            output["input_ids"] = output["input_ids"][:, :512]
            output["attention_mask"] = output["attention_mask"][:, :512]

        output=output.to(self.device)
        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = output.last_hidden_state[:, 0, :]

            feature = self.encoder(embedding)

            res = self.detector(feature)

        return res,feature

    def evaluate(self,loader):
        all_preds=[]
        all_labels=[]
        with tqdm(loader) as loop:
            for batch in loop:
                texts=batch["text"]
                labels=batch["label"]

                outputs,_= self.predict(texts)
                predictions = [0 if outputs[i][0] > outputs[i][1] else 1 for i in
                           range(len(outputs))]
                
                all_preds.extend(np.array(predictions))
                all_labels.extend(labels.cpu().numpy())
                acc=accuracy_score(all_labels,all_preds)
                loop.set_postfix({"acc":acc})
        f1=f1_score(all_labels,all_preds)
        return {"acc":acc,"f1":f1}


class Raidar:
    def __init__(self,data_save_path=None):
        self.model = load("model/raidar/raidar_arxiv copy.joblib")
        self.prompt="Revise this with your best effort"
        self.cutoff_start = 0
        self.cutoff_end = 6000000
        self.ngram_num=4
        if data_save_path:
            self.data_save_path=data_save_path

    @staticmethod
    def tokenize_and_normalize(sentence):
        """Tokenization and normalization."""
        return [word.lower().strip() for word in sentence.split()]

    @staticmethod
    def extract_ngrams(tokens, n):
        """Extract n-grams from the list of tokens."""
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def common_elements(list1, list2):
        """Find common elements between two lists."""
        return set(list1) & set(list2)

    def sum_for_list(self,a,b):
        return [aa+bb for aa, bb in zip(a,b)]

    def get_rewrite_text(self,loader):
        rewrite_dict=[]
        for item in tqdm(loader):
            text=item['text'][0]
            prompt=f"{self.prompt}: {text}"
            response=llm_api(prompt,model_name='gpt-3.5')
            rewrite_dict.append({"input":text,f"{self.prompt}":response})
        with open(self.data_save_path,'w') as f:
            for data in rewrite_dict:
                line=json.dumps(data)
                f.write(line+"\n")
        print(f"Data saved at {self.data_save_path}!")
        return rewrite_dict
    
    def get_data_stat(self,data_json):
        total_len = len(data_json)
        for idxx, each in enumerate(data_json):
        
            original = each['input']

            raw = self.tokenize_and_normalize(each['input'])
            if len(raw)<self.cutoff_start or len(raw)>self.cutoff_end:
                continue
            else:
                print(idxx, total_len)

            statistic_res = {}
            ratio_fzwz = {}
            all_statistic_res = [0 for i in range(self.ngram_num)]
            cnt = 0
            whole_combined=''
            for pp in each.keys():
                if pp != 'common_features':
                    whole_combined += (' ' + each[pp])
                

                    res = self.calculate_sentence_common(original, each[pp])
                    statistic_res[pp] = res
                    all_statistic_res = self.sum_for_list(all_statistic_res, res)

                    ratio_fzwz[pp] = [fuzz.ratio(original, each[pp]), fuzz.token_set_ratio(original, each[pp])]
                    cnt += 1
        
            each['fzwz_features'] = ratio_fzwz
            each['common_features'] = statistic_res
            each['avg_common_features'] = [a/cnt for a in all_statistic_res]

            each['common_features_ori_vs_allcombined'] = self.calculate_sentence_common(original, whole_combined)

        return data_json


    def calculate_sentence_common(self, sentence1, sentence2):
        """Calculate common words and n-grams between two sentences."""
        tokens1 = self.tokenize_and_normalize(sentence1)
        tokens2 = self.tokenize_and_normalize(sentence2)

        # Find common words
        common_words = self.common_elements(tokens1, tokens2)

        # Find common n-grams (up to 3-grams for this example)
        number_common_hierarchy = [len(list(common_words))]

        for n in range(2, 5):  # 2-grams to 4-grams
            ngrams1 = self.extract_ngrams(tokens1, n)
            ngrams2 = self.extract_ngrams(tokens2, n)
            common_ngrams = self.common_elements(ngrams1, ngrams2)
            number_common_hierarchy.append(len(list(common_ngrams)))

        return number_common_hierarchy

    def get_feature_vec(self, data):
        all_list = []
        for each in data:
            try:
                raw = self.tokenize_and_normalize(each['input'][0])
                r_len = len(raw) * 1.0
            except:
                continue

            each_data_fea = []
            if len(raw) < self.cutoff_start or len(raw) > self.cutoff_end:
                continue

            # Normalize features by sentence length
            each_data_fea = [ind_d / r_len for ind_d in each['avg_common_features']]
            for ek in each['common_features'].keys():
                each_data_fea.extend([ind_d / r_len for ind_d in each['common_features'][ek]])

            each_data_fea.extend([ind_d / r_len for ind_d in each['common_features_ori_vs_allcombined']])

            for ek in each['fzwz_features'].keys():
                each_data_fea.extend(each['fzwz_features'][ek])

            all_list.append(np.array(each_data_fea))
        all_list = np.vstack(all_list)

        return all_list

    def evaluate(self,loader):
        if not os.path.exists(self.data_save_path):
            rewrite_data=self.get_rewrite_text(loader)
        else:
            rewrite_data=load_data(self.data_save_path)
        data=self.get_data_stat(rewrite_data)
        # 提取特征
        X = self.get_feature_vec(data)
        # 进行预测
        y_pred = self.model.predict(X)
        y_ture=np.ones(len(data))
        accuracy = accuracy_score(y_ture, y_pred)
        f1=f1_score(y_ture, y_pred)
        return {"acc":accuracy,"f1":f1}
