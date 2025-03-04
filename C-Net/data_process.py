import pandas as pd
from openai import OpenAI
import argparse

def data_generate(args):
    client = OpenAI(api_key=args.api_key)
    original=pd.read_csv(args.data_path)
    machine_text=[]
    # print(original)
    for index, row in original.iterrows():
        if index<11436:
            continue
        question = row['question']
        # print(question.type())
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": question+"Answer in 150 words or less."}
            ]
        )
        print(index,response.choices[0].message.content)
        machine_text.append(response.choices[0].message.content)
        if (index+1)%2==0:
            machine_df = pd.DataFrame({'machine_text': machine_text})
            r_path="file_"+str(index)+"_1.csv"
            machine_df.to_csv(r_path, index=False)
    machine_df=pd.DataFrame({'machine_text':machine_text})
    print(machine_df)
    original=pd.concat([original,machine_df],axis=1)
    original.to_csv(args.save_path, index=False)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='chatgpt/unfilter_full/en_train.csv')
    parser.add_argument('--save_path',type=str, default='chatgpt/unfilter_full/train.csv')
    parser.add_argument('--api_key',type=str, default='')
    args, unparsed = parser.parse_known_args()
    data_generate(args)
