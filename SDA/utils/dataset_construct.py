import pandas as pd
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt_constructor import llm_api
import json
from dataloader import load_data
from tqdm import tqdm

def get_prompt():
    df=pd.read_csv('../LLM-MS/data/output.csv')
    domains=["books"]
    data_path='books_data.csv'
    for domain in domains:
        data=df[df['domain']==domain]
        target_data=data[data['model']=='human']
        prompt_data=target_data.sample(n=1200, random_state=42)
        my_data=prompt_data[['title','generation']]
        my_data.to_json(f'./data/{domain}_data_human.jsonl',orient='records', lines=True,force_ascii=False)
        # with open(f'./data/{domain}_data_human.jsonl','a') as file:
        #     for index,item in my_data.iterrows():
        #         line=json.dumps(item)
        #         file.write(line+"\n")
        # prompt_data.to_csv(f'./data/{domain}_data_human.jsonl',index=False)
    # return None

def get_human():
    prompt=load_data('./data/generation_abstracts_train_data.jsonl')
    df=pd.read_csv('../LLM-MS/data/output.csv')
    domains=["books","news","abstracts"]
    data=df[df['domain']=='abstracts']
    target_data=data[data['model']=='human']
    print(target_data)
    human_data=target_data.iloc[:1000]
    print(human_data)
    human_data.to_json("human_abstract.jsonl", orient="records", lines=True, force_ascii=False)

def load(path):
    data=[]
    with open(path,'r',encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def get_answer(data_path,save_path):
    # data=[]
    # with open(data_path,'r') as file:
    #     for line in file
    # df=pd.read_csv(data_path)
    data= pd.read_json(data_path, lines=True, encoding='utf-8')
    data=data.iloc[574:]
    print(data)
    # for index,row in df.iterrows():
    for index,row in tqdm(data.iterrows(),total=len(data)):
        title=row['title']
        prompt=f"Write the body of a plot summary for a novel titled \"{title}\""
        answer=llm_api(prompt)
        data={'prompt':prompt,'human':row['generation'],'answer':answer}
        with open(save_path,'a',encoding='utf-8') as file:
            line=json.dumps(data, ensure_ascii=False)
            file.write(line+"\n")
    # return None

def main():
    # domain_lst=['abstracts']
    # for domain in domain_lst:
    #     path=domain+'_data.csv'
        # save_path=domain+'_data.jsonl'
        # data_path=os.path.join('data',path)
        # save_path=os.path.join('data','generation_'+save_path)
    # tmp=get_answer("data/qwen_generation_books_test_data.jsonl",'data/generation_books_test.jsonl')
    tmp=get_answer("data/qwen_generation_books_train_data.jsonl",'data/generation_books_train.jsonl')
    # get_prompt()
    # data=pd.read_json('data/books_data_human.jsonl',lines=True, encoding='utf-8')
    # train_data=data.iloc[:600]
    # test_data=data.iloc[600:800]
    # eval_data=data.iloc[800:]
    # train_data.to_json('data/qwen_generation_books_train_data.jsonl', orient="records", lines=True, force_ascii=False)
    # test_data.to_json('data/qwen_generation_books_test_data.jsonl', orient="records", lines=True, force_ascii=False)
    # eval_data.to_json('data/qwen_generation_books_eval_data.jsonl', orient="records", lines=True, force_ascii=False)
    # print(f'Successfully generated {domain} data!\n')
    # get_answer('data/books_data_human.jsonl','data/books_data_human.jsonl')


if __name__=='__main__':
    # parser=argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str, default='data/÷
    main()
    # get_human()