import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prompt_constructor import llm_api
from utils.dataloader import load_data
import json
from tqdm import tqdm

def paraphrase(text,model='gpt-3.5'):
    prompt=f'Please rephrase the following sentence while maintaining the original meaning.\n{text}\nDirectly output the result.'
    response=llm_api(prompt,model_name=model)
    return response

def main():
    path=[
    'data/qwen_abstracts_test_data_feature_False_0.jsonl',]
    for file in path:
        attacked_data=[]
        data=load_data(file)
        for item in tqdm(data):
            text=item['answer']
            attacked_text=paraphrase(text)
            attacked_data.append({"answer":attacked_text,"label":1})
        save_path="attacked_results/paraphrase/"+file[:-6].split("/")[1]+"_paraphrase_attack_time.jsonl"
        with open(save_path, 'w') as f:
            for i in attacked_data:
                line=json.dumps(i)
                f.write(line+"\n")
        print(f"{file} attacked!")

main()