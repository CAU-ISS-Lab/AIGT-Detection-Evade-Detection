import nltk
from nltk.corpus import wordnet
import random
from tqdm import tqdm
import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataloader import load_data
import json


# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

def get_synonyms(word):
    """ 获取给定单词的同义词 """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    synonyms.discard(word)  # 排除原始单词
    return list(synonyms)

def replace_synonyms(text, rate=0.2):
    """ 用同义词替换文本中的单词，替换比例为rate """
    words = nltk.word_tokenize(text)
    num_words = len(words)
    num_to_replace = int(num_words * rate)  # 根据比例计算需要替换的单词数
    
    # 选择要替换的单词
    words_to_replace = random.sample(words, num_to_replace)
    
    new_words = []
    for word in words:
        if word in words_to_replace:
            synonyms = get_synonyms(word)
            if synonyms:
                new_word = random.choice(synonyms)
                new_words.append(new_word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)
    
    # 拼接替换后的文本
    new_text = ' '.join(new_words)
    return new_text

def main():
    ratio=0.2
    path=['data/deepseek_abstracts_test_data_feature_False_0.jsonl',
    'data/llama3.3_abstracts_test_data_feature_False_0.jsonl',
    'data/qwen_abstracts_test_data_feature_False_0.jsonl']
    for file in path:
        attacked_data=[]
        data=load_data(file)
        for item in tqdm(data):
            text=item['answer']
            attacked_text=replace_synonyms(text,ratio)
            attacked_data.append({"answer":attacked_text,"label":1})
        save_path="attacked_results/syn_replace/"+file[:-6].split("/")[1]+"_syn_attack.jsonl"
        with open(save_path, 'w') as f:
            for i in attacked_data:
                line=json.dumps(i)
                f.write(line+"\n")
        print(f"{file} attacked!")

main()

