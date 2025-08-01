import json 
from utils.detector import FastdetectGPT,RobertaDetector
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import argparse
from utils.dataloader import load_data
from test import get_best_feature
from prompt_constructor import llm_api
from tqdm import tqdm

def main(args):
    data_path=args.data_path
    data=load_data(data_path)
    cur_feature=get_best_feature(args.feature_path)
    if args.detector=='chatgptdetector':
        detector=RobertaDetector('cuda:0',True)
    else:
        detector=FastdetectGPT('gpt2','gpt2')
    answer_path=f'RAG_data/qwen_abstracts_train_data_{args.detector}_RAG_MPU.jsonl'
    epoch=5
    for i in range(0,epoch):
        fail_answer=[]
        for item in tqdm(data):
            prompt=item["prompt"]
            cur_prompt='The features of texts that closely resemble human writing are as follows: \n{}.\n Question: {}. Directly output the answer in human style.'.format(cur_feature,prompt)
            answer=llm_api(cur_prompt,model_name='qwen')
            pred_class,prob=detector(answer)
            if pred_class==0:
                with open(answer_path,'a',encoding='utf-8') as file:
                    line=json.dumps({"prompt":prompt,"answer":answer})
                    file.write(line+"\n")
            else:
                fail_answer.append({"prompt":prompt})
        data=fail_answer
        if not fail_answer:
            print("data generated!")
            break
    print(f"fail data lenth: {len(fail_answer)}")
    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,default='data/abstracts_train_data.jsonl')
    parser.add_argument('--save_data_dir',type=str,default='feature')
    parser.add_argument('--data_generator',type=str,default='qwen')
    parser.add_argument('--feature_path',type=str,default='feature/qwen_optimized_best_feature_repeat_abstracts_1_new_MPU.json')
    parser.add_argument('--detector',type=str,default='chatgptdetector',choices=['chatgptdetector','fastdetect'])
    args,unparsed=parser.parse_known_args()
    main(args)

    
