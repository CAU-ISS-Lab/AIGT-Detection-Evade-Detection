from openai import OpenAI
import os
from tqdm import tqdm
import torch
import json
import requests

model={"deepseek":"deepseek-v3","llama3.3":"meta-llama/Llama-3.3-70B-Instruct",'qwen':"qwen-max",'gpt-3.5':'gpt-3.5-turbo'}

def llm_api(prompt,system_prompt="You are a helpful assistant.",model_name='qwen'):
    response_dict={}
    if model_name!='llama3.3':
        client=OpenAI(
            api_key=os.environ.get("API_KEY"),
            base_url=os.environ.get("BASE_URL")
        )
        while not hasattr(response_dict, 'choices') or not response_dict.choices:
            response_dict = client.chat.completions.create(
                model=model[model_name],
                messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                ],
                stream=False
                )
        return response_dict.choices[0].message.content
    else:
        while "choices" not in response_dict:
            url = "https://api.siliconflow.cn/v1/chat/completions"
            payload = {
                "model": model[model_name],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,}
            headers = {
                "Authorization": "",
                "Content-Type": ""}
            response = requests.request("POST", url, json=payload, headers=headers)
            print(response.text)
            response_dict=json.loads(response.text)
        return response_dict["choices"][0]["message"]["content"]
    
def example_prompt_constructor(successfully_escape):
    # example_prompt='Here are some questions and their corresponding machine-generated answers, which are considered to be very close to human-written text: \n'
    example_prompt='Here are some machine-generated answers, which are considered to be very close to human-written text: \n'
    for index,item in enumerate(successfully_escape):
        # example='Question: {}\n Answer:{}\n\n'.format(item['prompt'],item['answer'])
        example='text {}: {}\n'.format(index,item['answer'])
        example_prompt+=example
    # example_prompt+='\nHere are some machine-generated answers, which are considered to significantly different from human-written text: \n'
    # for index,item in enumerate(fail_escape):
    #     # example='Question: {}\n Answer:{}\n\n'.format(item['prompt'],item['answer'])
    #     example='text {}: {}\n'.format(index,item['answer'])
    #     example_prompt+=example
    return example_prompt


def feature_construct(successfully_escape,cur_feature=''):
    example_prompt=example_prompt_constructor(successfully_escape)
    if cur_feature!='':
          # The machine generates replies that resemble human responses based on the following text characteristics: \n{}\n\
        #  These features should include syntactic structure, grammatical structure, language style, punctuation usage, vocabulary choice, etc., without considering the text semantics and themes. \
        optimize_prompt='{}\n\
        The current features of texts that closely resemble human writing are as follows:\n{}\n\
         Based on the examples above, refine the current features to make the generated responses more closely resemble human replies.\
        \nDirectly output the revised features in following format(do not show specific example):\n\
        **syntactic structure**\n\
        feature\n\
        **grammatical structure**\n\
        feature\n ...'.format(example_prompt,cur_feature)
        feature=llm_api(optimize_prompt)
    else:
        construct_prompt='Based on the examples above, summarize the characteristics of these texts that closely resemble human writing, \
        including syntactic structure, grammatical structure, language style, punctuation usage, vocabulary choice, etc., without considering the text semantics and themes.\n \
        Directly output the summarized features in following format(do not show specific example):\n\
        **syntactic structure**\n\
        feature\n\
        **grammatical structure**\n\
        feature\n ...'
        prompt='{}{}'.format(example_prompt,construct_prompt)
        feature=llm_api(prompt)
    return feature

def feature_optimize(train_loader,epoch_num,detector,cur_feature='',repeat=5,optimize_step=5,error_range=1):
    batch_success_es=[]
    batch_fail_es=[]
    stop=0
    for epoch in range(0,epoch_num):
        init_idx=0
        stop=0
        fail_escape=[]
        successfully_escape=[]
        for index,data in enumerate(tqdm(train_loader)):
            # for data in loop:
            prompt=data['prompt']
            if cur_feature:
                cur_prompt='The features of texts that closely resemble human writing are as follows: \n{}.\n Question: {}. Directly output the answer in human style.'.format(cur_feature,prompt)
            else:
                cur_prompt=f"Question: {prompt}. Directly output the answer in human style."
            for i in range(repeat):
                answer=llm_api(cur_prompt)
                pred_class,pred_prob=detector(answer)
                if pred_class == 0:
                    batch_success_es.append({'prompt':prompt,'answer':answer})
                    print('sucess escape ',len(batch_success_es))
                    break
            if pred_class!=0:
                fail_escape.append({'prompt':prompt,'answer':answer})
                batch_fail_es.append({'prompt':prompt,'answer':answer})
            if len(batch_success_es)>=optimize_step:
                step_range=index-init_idx
                init_idx=index
                with open(f"feature/qwen_optimize_feature_repeat_abstracts_{repeat}_new_MPU.jsonl",'a',encoding='utf-8') as json_file:
                    bat_dict={'epoch':epoch,'step_range':step_range,"cur_feature":cur_feature}
                    line=json.dumps(bat_dict)
                    json_file.write(line+'\n')
                print("Optimized feature saved!")
                if step_range-optimize_step<=error_range:
                    stop+=1
                    print("stop: ",stop)
                else:
                    stop=0
                    print("features optimizing....")
                    cur_feature=feature_construct(batch_success_es,cur_feature)
                    print("done!")
                successfully_escape.extend(batch_success_es)
                batch_success_es=[]
                batch_fial_es=[]
            if stop>2:
                print('feature optimized!')
                break
        with open(f"feature/qwen_successfully_repeat_abstracts_{repeat}_new_MPU.jsonl",'a',encoding='utf-8') as json_file:
            for item in successfully_escape:
                line=json.dumps(item)
                json_file.write(line+'\n')
        if stop>2:
            with open(f"feature/qwen_optimized_best_feature_repeat_abstracts_{repeat}_new_MPU.json",'w',encoding='utf-8') as json_file:
                bat_dict={'epoch':epoch,'step_range':step_range,"cur_feature":cur_feature}
                json.dump(bat_dict,json_file)
            break
        train_loader=fail_escape
    return cur_feature


            # orig_answer=data['answer']
            # orig_class,orig_prob=detector(orig_answer)
            # if orig_class!=0:
            #     fail_escape.append({'prompt':prompt,'answer':orig_answer})
            # else:
            #     successfully_escape.append({'prompt':prompt,'answer':orig_answer})
            # prompt='{}\n{}'.format(prompt,cur_feature)
            # for i in range(repeat):
            #     regeneration=llm_api(prompt)
            #     pred_class,prob=detector(regeneration)
            #     if prob[1]>0.5:

            #feature opimize
            

            
