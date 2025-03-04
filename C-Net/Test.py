from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import torch.optim as optim
import argparse
from transformers import RobertaModel,RobertaTokenizer,AutoTokenizer,AutoModel,GPT2Model,GPT2Tokenizer
from framework import CNetModel
from torch.utils.data import  DataLoader
from tqdm import tqdm
from dataloader import get_train_data,get_test_data
import torch
from Train import setup_distributed
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import os
os.environ["OMP_NUM_THREADS"] = "2"
def tsne_plot(x1,x2,label):
    tsne=TSNE(n_components=2)
    Y1=tsne.fit_transform(x1)
    print(label)
    machine= [index for index, value in enumerate(label) if value == 1]
    human=[index for index, value in enumerate(label) if value == 0]
    # print(machine)
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    # print(Y1)
    tsne_machine=Y1[machine]
    tsne_human=Y1[human]
    # print(tsne_machine)
    plt.scatter(tsne_machine[:, 0], tsne_machine[:, 1], 1, color="purple")
    plt.scatter(tsne_human[:, 0], tsne_human[:, 1], 1, color="green")
    # plt.legend(loc='upper left')
    plt.title("Original", x=0.5, y=-0.3)
    # plt.show()

    plt.subplot(122)
    Y2=tsne.fit_transform(x2)
    tsne_machine = Y2[machine]
    tsne_human = Y2[human]
    plt.scatter(tsne_human[:, 0], tsne_human[:, 1], 1, color="green")
    plt.scatter(tsne_machine[:, 0], tsne_machine[:, 1], 1, color="purple")
    plt.savefig('Visualization_sent.png')
    plt.title("C-Net", x = 0.5, y = -0.3)
    plt.show()


def main(args):
    rank, world_size = setup_distributed()
    if args.device is None:
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    print('rank:', rank, 'world_size:', world_size, 'device:', device)
    print("Fetching data...")
    test_data_path = args.data_dir + args.test_data_file
    test_dataset = get_test_data( test_data_path)
    if "gpt" in args.pretrained_model:
        tokenizer = GPT2Tokenizer.from_pretrained(args.pretrained_model)
        pre_model = GPT2Model.from_pretrained(args.pretrained_model)
    else:
        if any(plm if plm in args.pretrained_model else False for plm in ["bart","electra","chinese"]):
            print("1")
            tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
            pre_model = AutoModel.from_pretrained(args.pretrained_model)
        else:
            print("2")
            tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model)
            pre_model = RobertaModel.from_pretrained(args.pretrained_model)
    pre_model.requires_grad_(False)

    model = CNetModel(pre_model,tokenizer)
    model_path=args.model+".pth"
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True)

    for epoch in range(args.epoch):
        model.eval()
        all_predictions = []
        all_targets = []
        embeddings=[]
        all_outputs=[]

        for sample_batch in tqdm(test_loader, desc=f'Epoch {epoch + 1}/{args.epoch}'):
            inputs = sample_batch["text"]
            targets = sample_batch["label"]
            outputs,embedding = model.predict(inputs, device)
            predictions = [0 if outputs[i][0] > outputs[i][1] else 1 for i in range(len(outputs))]

            all_predictions.extend(np.array(predictions))
            all_targets.extend(targets.cpu().numpy())
            embeddings.append(embedding.cpu().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())

            batch_accuracy = accuracy_score(all_targets, all_predictions)
            print(f'Acc: {batch_accuracy:.4f}')

        # embeddings = np.concatenate(embeddings, axis=0)
        # all_outputs = np.concatenate(all_outputs, axis=0)
        # tsne_plot(embeddings,all_outputs,all_targets)
        accuracy = accuracy_score(all_targets, all_predictions)
        auroc = roc_auc_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)

        print(
            f"Epoch [{epoch + 1}/{args.epoch}] Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='chatgpt/unfilter_full/')
    parser.add_argument('--test_data_file', type=str, default='test.csv')
    parser.add_argument('--optimizer',type=str,default='Adam')
    parser.add_argument('--lr',type=float,default=0.0005)
    parser.add_argument('--epoch',type=int,default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device',type=str,default=None)
    parser.add_argument('--model',type=str,default="model/model_tmp")
    parser.add_argument('--pretrained_model', type=str, default="facebookAI/roberta-base")
    args, unparsed = parser.parse_known_args()
    main(args)