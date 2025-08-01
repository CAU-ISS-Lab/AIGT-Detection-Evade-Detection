import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataloader import load_data
from utils.detector import RobertaDetector,RADAR,test_dataset,FastdetectGPT,Raidar,NPnet
import argparse
from transformers import RobertaTokenizer,RobertaModel
from torch.utils.data import DataLoader
import json
import torch

def main(args):
    device='cuda:1' if torch.cuda.is_available() else 'cpu'
    if args.detector=='chatgptDetector':
        print(args.detector)
        detector=RobertaDetector(device)
    elif args.detector=='radar':
        detector=RADAR(device)
    elif args.detector=='mpu':
        detector=RobertaDetector(device,True)
    elif args.detector=='raidar':
        data_type=args.data_path.split("/")[-1][:-6]
        data_save_path=f"attacked_results/raidar/{data_type}_rewrite.jsonl"
        detector=Raidar(data_save_path)
    elif args.detector=='npnet':
        detector=NPnet(device,'gaussian')
        detector.load_state_dict(torch.load('model/npnet/text_g_30.pth',map_location=device))
        detector.to(device)
    else :
        detector=FastdetectGPT('../model/gpt2','../model/gpt2')
    
    print(type(args.data_path))
    data=load_data(args.data_path)
    dataset=test_dataset(data)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True)
    result=detector.evaluate(loader)
    print(f'======human vs {args.target} using {args.detector}======')
    print(result)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--detector',type=str,choices=['chatgptDetector','radar','fast-detect','mpu','raidar','npnet'])
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--data_path',type=str,default='data/qwen_abstracts_test_data_feature_True_5_chatgptDetector_True_MPU.jsonl')
    parser.add_argument('--target',type=str,default='chatgpt_answers')
    args,unparsed=parser.parse_known_args()
    main(args)