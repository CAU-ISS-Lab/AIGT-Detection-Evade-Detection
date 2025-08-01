import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataloader import load_data
from transformers import AutoTokenizer,AutoModel,AutoModelForCausalLM
from torch.utils.data import DataLoader
from detector import test_dataset
from tqdm import tqdm
import torch
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon, cosine
from scipy.special import kl_div
from sklearn.decomposition import PCA
import nltk
from nltk.translate.bleu_score import sentence_bleu
import seaborn as sns
from rouge import Rouge
from prompt_constructor import llm_api
import re
from bert_score import score

def extract_feature(data,model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval() 

    features = [] 

    for item in tqdm(data):
        text = item["text"] 
        encoded_text = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        encoded_text.to(device)

        with torch.no_grad():
            outputs = model(**encoded_text)

        last_hidden_state = outputs.last_hidden_state
        sentence_embedding = last_hidden_state.detach().cpu().mean(dim=1).squeeze().numpy()

        features.append(sentence_embedding)

    return np.vstack(features)

    

def tsne_visual(all_embedding,save_path,labels,model_path,mode='pca'):
    tsne = TSNE(n_components=1, perplexity=30, random_state=42)
    pca=PCA(n_components=1)
    # plt.figure(figsize=(8, 6))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # ax = fig.add_subplot(111, projection='3d')
    i=0
    colors = ["#F4DDB6", 
    # "#F5CABC", 
    # "#F4B4BE",
    "#CEA2B5",
    # "#9C9CC0",
    # "#6F8BB8",
    '#59649D']
    subplot_idx=0
    llm_name=['Qwen-max','Llama3.3-70B-insturct','Deepseek-v3']
    for llm_embedding in all_embedding.values():
        i=0
        all_feature={}
        index=0
        for embedding in llm_embedding.values():
            if mode=='pca':
                print(f"Extract data {index} pca feature...")
                pca_feature=pca.fit_transform(embedding)
                all_feature[index]=pca_feature
            else:
                print(f"Extract data {index} tsne feature...")
                tsne_feature=tsne.fit_transform(embedding)
                all_feature[index]=tsne_feature
            index+=1
        print("All feature constructed!")
        print("visualizing...")
        for feature in all_feature.values():
        # plt.scatter(feature[:,0],feature[:,1],label=labels[i],alpha=0.7, s=40)
        # i+=1
        # plt.figure(figsize=(6, 4))
        # print(feature.shape)
            sns.kdeplot(feature[:,0],ax=axes[subplot_idx],color=colors[i], fill=True, alpha=0.5, label=labels[i])
            i+=1
        axes[subplot_idx].set_title(llm_name[subplot_idx], y=-0.2,fontsize=18)
        axes[subplot_idx].legend(fontsize=14)
        axes.flat[subplot_idx].set_ylabel('Density',fontsize=14)
        subplot_idx+=1
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path)
    return all_feature

# def density_visual=(all_feature,labels):
#     for feature in all_feature.values():


def JS_div(P,Q):
    M=0.5*(P+Q)
    return 0.5 * (entropy(P, M) + entropy(Q, M))

def cal_cos_sim(features):
    cos_sim_lst=[]
    Q=features[len(features)-1]

    for dist_idx in range(0,len(features)-1):
        P=features[dist_idx]
        cos_lst_tmp=[]
        for i in range(P.shape[0]):
            P_norm=(P[i])/np.sum((P[i]))
            Q_norm=(Q[i])/np.sum((Q[i]))

            cos_lst_tmp.append(1 - cosine(P_norm, Q_norm))
    
        cos_sim_lst.append(np.mean(cos_lst_tmp))

    return cos_sim_lst

def cal_ppl(loader,model_name):
    total_loss=0
    total_tokens=0
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForCausalLM.from_pretrained(model_name)
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval() 
    for item in tqdm(loader):
        text=item["text"]
        encoded_text=tokenizer(text,truncation=True,max_length=512,return_tensors='pt')
        encoded_text.to(device)

        with torch.no_grad():
            outputs=model(**encoded_text,labels=encoded_text["input_ids"])
            loss = outputs.loss

            num_tokens = encoded_text["input_ids"].numel()

            total_loss += loss.item() * num_tokens  
            total_tokens += num_tokens  

    avg_loss = total_loss / total_tokens 
    avg_ppl = torch.exp(torch.tensor(avg_loss)).item()  
    return avg_ppl


def self_bleu(sentences, n_gram=4):
    if len(sentences) < 2:
        raise ValueError("Self-BLEU 需要至少 2 个文本")

    bleu_scores = []
    
    for i, candidate in enumerate(sentences):
        references = [sent.split() for j, sent in enumerate(sentences) if j != i]
        candidate_tokens = candidate.split()

        bleu_score = sentence_bleu(references, candidate_tokens, weights=[1/n_gram] * n_gram)
        bleu_scores.append(bleu_score)
    
    return np.mean(bleu_scores)

def distinct_n(texts, n=1):
    ngrams_list = []
    total_ngrams = 0

    for text in texts:
        words = text.split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)] 
        total_ngrams += len(ngrams)
        ngrams_list.extend(ngrams)

    unique_ngrams = len(set(ngrams_list)) 
    return unique_ngrams / total_ngrams if total_ngrams > 0 else 0

def cal_rouge( generated_texts,reference_texts):
    rouge = Rouge()
    
    total_rouge_1 = 0
    total_rouge_2 = 0
    total_rouge_L = 0
    num_samples = len(reference_texts)
    
    for reference, generated in zip(reference_texts, generated_texts):
        # print(reference,generated)
        if not reference.strip() or not generated.strip():
            # print(f"Warning: Skipping empty reference or generated text pair.")
            continue
        scores = rouge.get_scores(generated, reference)[0] 
        total_rouge_1 += scores["rouge-1"]["f"]
        total_rouge_2 += scores["rouge-2"]["f"]
        total_rouge_L += scores["rouge-l"]["f"]
    
    # 计算平均值
    avg_rouge_1 = total_rouge_1 / num_samples
    avg_rouge_2 = total_rouge_2 / num_samples
    avg_rouge_L = total_rouge_L / num_samples
    
    return {"rouge_1":avg_rouge_1, "rouge_2":avg_rouge_2, "rouge_L":avg_rouge_L}

def cal_gptscore(texts):
    all_score=0
    for text in tqdm(texts):
        res='aaaaaa'
        while bool(re.search(r'[^0-9.]', text)):
            prompt=f'Consider the four aspects of grammar usage, coherence, fluency, and readability, and rate the following text on a scale from 1 to 10:\n\
            text: {text} \n \
            Directly output the overall score in following format:\n\
            Overall Score: 0.0 '
            score=llm_api(prompt,model_name='gpt-3.5')
            print(score)
            res=score.split(":")[-1]
        all_score+=float(res)
    avg_score=all_score/len(texts)
    return avg_score

def cal_bertscore(generated_texts,human_texts):
    all_f1=0
    p,r,f1=score(generated_texts,human_texts,model_type='roberta-base')
    
    # 计算平均值
    all_f1  = f1.mean().item()
    
    return all_f1

def text_quantity_visual():
    s

    
def main():
    llms='qwen'
    model_path='model/roberta-base'
    mode='pca'
    model=model_path.split("/")[-1]
    is_ppl=False
    is_visual=False
    is_cos=False
    is_sbleu=False
    is_distinct=False
    is_ROUGE=False
    is_gptscore=False
    is_bertscore=True
    labels=['Direct Generation',
    # 'paraphrase',
    # 'SICO',
    # 'HMGC',
    # 'dipper',
    'SDA',
    'human']
    all_embedding={}
    for llm in ['qwen','llama3.3','deepseek']:
        data_lst=[f'data/{llm}_abstracts_test_data_feature_False_0.jsonl',
    # f'attacked_results/paraphrase/{llm}_abstracts_test_data_feature_False_0_paraphrase_attack.jsonl',
    # f'attacked_results/SICO/generated_text_{llm}.tsv',
    # f'attacked_results/HMGC/{llm}_HMGC.jsonl',
    # f'attacked_results/dipper/{llm}_abstracts_test_data_pp.jsonl',
    f'data/{llm}_abstracts_test_data_feature_True_5_chatgptDetector_True_answer_only.jsonl',
    'data/human_abstracts_test.jsonl']
        if is_ROUGE or is_bertscore:
            human_data=load_data(data_lst[-1])
            human=[item['answer'] for item in human_data]
            # print(human)
    # save_path=f'visual/{llm}_{model}_{mode}_density.png'
        save_path=f'visual/{model}_{mode}_density_1.png'
        llm_embedding={}
        all_ppl=[]
        all_self_bleu=[]
        all_distinct_n=[]
        all_rouge=[]
        all_score=[]
        all_bertscore=[]
        for index,data_path in enumerate(data_lst):
            # print(f"Extract data {index} feature...")
            data=load_data(data_path)
            dataset=test_dataset(data)
            loader=DataLoader(dataset,batch_size=1,shuffle=True)
            all_text=[item["answer"] for item in data]
            if is_gptscore:
                score=cal_gptscore(all_text)
                all_score.append(score)
            if is_bertscore:
                bertscore=cal_bertscore(all_text,human)
                all_bertscore.append(bertscore)
            if is_ROUGE:
                # print(f"Calculate rouge of data {index}...")
                rouge_data=cal_rouge(all_text,human)
                all_rouge.append(rouge_data)
            if is_ppl and index!=len(data_lst)-1:
                print(f"Calculate  perplexity of data {index}...")
                ppl=cal_ppl(loader,model_path)
                all_ppl.append(ppl)
            if is_sbleu and index!=len(data_lst)-1:
                print(f"Calculate  self-BLEU of data {index}...")
                all_self_bleu.append(self_bleu(all_text))
            if is_distinct and index!=len(data_lst)-1:
                print(f"Calculate  self-BLEU of data {index}...")
                all_distinct_n.append(distinct_n(all_text))
        #     embedding=extract_feature(loader,model_path)
        #     llm_embedding[index]=embedding
        # print(f"Embedding extracted!")
        # all_embedding[llm]=llm_embedding

        if is_ppl:
            print(f"********{llm} Perplexity********")
            for idx,label in enumerate(labels):
                if label=='human':
                    continue
                print(f"{label}:{all_ppl[idx]}")
            print("*************************")

        if is_sbleu:
            print(f"********{llm} sbleu********")
            for idx,label in enumerate(labels):
                if label=='human':
                    continue
                print(f"{label}:{all_self_bleu[idx]}")
            print("*************************")
    
        if is_distinct:
            print(f"********{llm} distinct_n********")
            for idx,label in enumerate(labels):
                if label=='human':
                    continue
                print(f"{label}:{all_distinct_n[idx]}")
            print("*************************")

        if is_ROUGE:
            print(f"********{llm} Rouge********")
            for idx,label in enumerate(labels):
                if label=='human':
                    continue
                print(f"{label}:{all_rouge[idx]}")
            print("*************************")
        
        if is_cos :
            print("Calculate cosine similarity...")
            cos_res=cal_cos_sim(all_embedding)
            print(f"********{llm} Perplexity********")
            for idx,label in enumerate(labels):
                if label=='human':
                    continue
                print(f"{label}:{cos_res[idx]}")
            print("*************************")
        
        if is_gptscore :
            print(f"********{llm} score********")
            for idx,label in enumerate(labels):
                if label=='human':
                    continue
                print(f"{label}:{all_score[idx]}")
            print("*************************")
        
        if is_bertscore :
            print(f"********{llm} bertscore********")
            for idx,label in enumerate(labels):
                if label=='human':
                    continue
                print(f"{label}:{all_bertscore[idx]}")
            print("*************************")



    if is_visual:
        features=tsne_visual(all_embedding,save_path,labels,model_path,mode)

if __name__=='__main__':
    main()

        

    
