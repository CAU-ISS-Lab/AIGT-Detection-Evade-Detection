from utils.detector import RobertaDetector,FastdetectGPT
import argparse
from prompt_constructor import feature_optimize
import os
import json
import time

def main(args):
    st=time.time()
    if args.detector=='chatgptDetector':
        detector=RobertaDetector('cuda:0',True)
    else:
        detector=FastdetectGPT('gpt2','gpt2')
    data=[]
    with open(args.train_data_path,'r', encoding='utf-8') as file:
        for line in file:
            line_data=json.loads(line)
            data.append(line_data)
    feature=feature_optimize(data,args.epoch,detector,repeat=args.repeat,optimize_step=args.optimize_step,error_range=args.error_range)
    et=time.time()
    print(et-st)
    print(feature)
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--train_data_path',type=str,default='data/generation_abstracts_train_data.jsonl')
    parser.add_argument('--epoch',type=int,default=5)
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--optimize_step',type=int,default=5)
    parser.add_argument('--error_range',type=int,default=1)
    parser.add_argument('--detector',type=str,default='chatgptDetector')
    args,unparsed=parser.parse_known_args()
    main(args)