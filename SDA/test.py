import torch
from utils.detector import RobertaDetector,test_dataset,FastdetectGPT
import argparse
from utils.dataloader import load_data,dataset
from prompt_constructor import llm_api,example_prompt_constructor
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import os
from RAG import FaissRAG
import time

def get_best_feature(feature_path):
    # with open(feature_path,'r',encoding='utf-8') as file:
    #     data=json.load(file)
    data=load_data(feature_path)
    return data[-1]['cur_feature']

def main(args):
    print(args.is_RAG)
    if args.detector=='chatgptDetector':
        detector=RobertaDetector('cuda:0')
    else:
        detector=FastdetectGPT('gpt2','gpt2')
    data=load_data(args.data_path)
    # data=data[:50]
    index_path=f"index/index_{args.detector}_{args.llm}_{args.example_size}_MPU.index"
    repeat=args.repeat
    if args.optimized_feature:
        # feature_path=f'feature/qwen_optimized_best_feature_repeat_1.json'
        feature_path=f'feature/qwen_optimized_best_feature_repeat_abstracts_1_new_MPU.json'
        cur_feature=get_best_feature(feature_path)
    else:
        cur_feature=''
    print("**current feature**\n",cur_feature)
    print("****get answer****")
    answer_path=f'data/{args.llm}_{args.domain}_test_data_feature_{args.optimized_feature}_{args.example_size}_{args.detector}_{args.is_RAG}_MPU.jsonl'
    # answer_path=f'data/{args.llm}_{args.domain}_test_data_feature_{args.optimized_feature}_{args.example_size}_{args.detector}_{args.is_RAG}.jsonl'
    if args.is_RAG:
        RAG=FaissRAG('cuda:0',args.example_data_path,index_path,'../model/gpt2')
    if os.path.exists(answer_path):
        test_data=load_data(answer_path)
    else:
        test_data=[]
        if args.example_size!=0 and args.is_RAG==False:
            print("example construct...")
            example_data=load_data(args.example_data_path)
            example_prompt=example_prompt_constructor(example_data[:args.example_size])
            # print(example_prompt)
        else:
            example_prompt=''
        st=time.time()
        for item in tqdm(data):
            prompt=item["prompt"]
            if args.is_RAG:
                rag_st=time.time()
                result_ICL=RAG.search(prompt,args.example_size)
                rag_et=time.time()
                # print(rag_et-rag_st)
                example_prompt=example_prompt_constructor(result_ICL)
            if cur_feature:
                cur_prompt='{}\n The features of texts that closely resemble human writing are as follows: \n{}. Question: {}. Directly output the answer in human style.'.format(example_prompt,cur_feature,prompt)
            else:
                cur_prompt=f"{example_prompt}\n Question: {prompt}. Directly output the answer in human style."
            answer=llm_api(cur_prompt,model_name=args.llm)
            if 'train' not in args.data_path:
                test_data.append({"answer":answer,"label":1})
            else:
                test_data.append({"prompt":prompt,"answer":answer,"label":1})
        et=time.time()
        print('generation time', et-st)
        with open(answer_path,'w',encoding='utf-8') as file:
            for item in test_data:
                line=json.dumps(item)
                file.write(line+'\n')
    print("start evaluating...")
    test_datasets=test_dataset(test_data)
    test_loader=DataLoader(test_datasets,batch_size=args.test_batch_size,shuffle=True)
    res_dict=detector.evaluate(test_loader)
    print(res_dict)
    save_res_path=args.save_res_path+f'{args.llm}_{args.domain}_test_data_{args.optimized_feature}_{args.example_size}_{args.detector}_{args.is_RAG}_answer_only.json'
    # save_res_path=args.save_res_path+f'{args.llm}_{args.domain}_test_data_{args.optimized_feature}_{args.example_size}_{args.detector}_{args.is_RAG}.json'
    with open(save_res_path,'w',encoding='utf-8') as file:
        json.dump(res_dict,file)
    print(f"results saved at {save_res_path}")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str,default='data/generation_abstracts_test_data.jsonl')
    parser.add_argument('--example_data_path',type=str,default='RAG_data/qwen_abstracts_train_data_chatgptdetector_RAG_MPU.jsonl')
    parser.add_argument('--save_res_path',type=str,default='results/')
    parser.add_argument('--domain',type=str,default='abstracts')
    parser.add_argument('--optimized_feature',action='store_true')
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--example_size',type=int,default=5)
    parser.add_argument('--test_batch_size',type=int,default=1)
    parser.add_argument('--llm',type=str,default='qwen')
    parser.add_argument('--detector',type=str,default='chatgptDetector')
    parser.add_argument('--is_RAG', action='store_true')
    # parser.add_argument('--error_range',type=int,default=2)
    args,unparsed=parser.parse_known_args()
    main(args)

