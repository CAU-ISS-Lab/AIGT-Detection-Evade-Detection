import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import torch
import argparse
from transformers import RobertaModel,RobertaTokenizer
from framework import RLADmodel,RLADloss
from dataloader import get_data
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

def fgsm_attack(image, epsilon, data_grad):
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = data_grad.sign()
    # 通过epsilon生成对抗样本
    perturbed_image = image + epsilon*sign_data_grad
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回对抗样本
    return perturbed_image

def main(args):
    if args.device is None:
        device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
    print("Fetching data...")

    test_data_path = args.data_dir + args.test_data_file
    test_dataset = get_data( test_data_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.pretrained_model)
    pre_model = RobertaModel.from_pretrained(args.pretrained_model)
    pre_model.requires_grad_(False)
    model = RLADmodel(tokenizer,pre_model,device,'')

    model_path=args.model+".pth"
    state_dict = torch.load(model_path,map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params / 1000000)
    test_loader=DataLoader(test_dataset,batch_size=1,shuffle=True)
    loss_fn = RLADloss(args.weight_1, args.weight_2, args.weight_3)

    for epoch in range(args.epoch):
        model.eval()
        all_predictions = []
        all_adv_pred=[]
        all_targets = []
        all_outputs=[]
        data=[]
        start_time = time.time()
        with tqdm(test_loader, desc=f'Epoch {epoch + 1}/{args.epoch}') as loop:
            for sample_batch in loop:
                inputs = sample_batch["text"]
                targets = sample_batch["label"].to(device)
                # print(targets)
                outputs = model.predict(inputs)
                predictions = [0 if outputs[i][0] > outputs[i][1] else 1 for i in
                               range(len(outputs))]

                # Collect predictions and targets for metrics calculation
                all_predictions.extend(np.array(predictions))
                all_targets.extend(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().detach().numpy())

                batch_accuracy = accuracy_score(all_targets, all_predictions)
                loop.set_postfix(acc=batch_accuracy)

            accuracy = accuracy_score(all_targets, all_predictions)
            auroc = roc_auc_score(all_targets, all_predictions)
            precision = precision_score(all_targets, all_predictions)
            recall = recall_score(all_targets, all_predictions)
            f1 = f1_score(all_targets, all_predictions)

            print(
                f"Epoch [{epoch + 1}/{args.epoch}] Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='./')
    parser.add_argument('--test_data_file', type=str, default='M4/arxiv_chatGPT_test.jsonl')
    parser.add_argument('--weight_1', type=float, default=0.5)
    parser.add_argument('--weight_2', type=float, default=1)
    parser.add_argument('--weight_3', type=float, default=0.01)
    parser.add_argument('--optimizer',type=str,default='Adam')
    parser.add_argument('--lr',type=float,default=0.0005)
    parser.add_argument('--epoch',type=int,default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device',type=str,default=None)
    parser.add_argument('--model',type=str,default="./encoder/uniform_4000_30")
    parser.add_argument('--pretrained_model', type=str, default="roberta-base")
    args, unparsed = parser.parse_known_args()
    main(args)