import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import torch.optim as optim
import torch
import argparse
from transformers import RobertaModel,RobertaTokenizer,AutoTokenizer,AutoModel,GPT2Model,GPT2Tokenizer
from framework import CNetModel,CNetLoss
from dataloader import get_train_data,get_test_data
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import torch.distributed as dist

def setup_distributed(port=29500):
    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        return 0, 1

    if 'MPIR_CVAR_CH3_INTERFACE_HOSTNAME' in os.environ:
        from mpi4py import MPI
        mpi_rank = MPI.COMM_WORLD.Get_rank()
        mpi_size = MPI.COMM_WORLD.Get_size()

        os.environ["MASTER_ADDR"] = '127.0.0.1'
        os.environ["MASTER_PORT"] = str(port)

        dist.init_process_group(backend="nccl", world_size=mpi_size, rank=mpi_rank)
        return mpi_rank, mpi_size

    dist.init_process_group(backend="nccl", init_method="env://")
    return dist.get_rank(), dist.get_world_size()

def train(args,model,optimizer,train_loader,criterion,weight_1,weight_sim,weight_ncd,epoch,device):
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_targets = []

    for sample_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epoch}'):
        optimizer.zero_grad()
        machine_text=sample_batch["machine_input"]
        inputs = sample_batch["text"]
        targets=sample_batch["label"].to(device)
        # print(machine_text)
        print(len(inputs),type(inputs))
        outputs, similarity,ncd_distance = model(machine_text,inputs,device)

        loss,outputs = criterion(outputs, targets,similarity, ncd_distance, weight_1,weight_sim,weight_ncd,device)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() *len(inputs)
        # print(outputs)
        predictions = [0 if outputs[i][0] > outputs[i][1] else 1 for i in range(len(outputs))]

        all_predictions.extend(np.array(predictions))
        all_targets.extend(targets.cpu().numpy())

        batch_accuracy = accuracy_score(all_targets, all_predictions)
        current_loss = loss.item()
        print(f'Acc: {batch_accuracy:.4f} Loss: {current_loss:.4f}')

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    auroc = roc_auc_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    return {
        "AUROC":auroc,
        "Acc":accuracy,
        "precision":precision,
        "recall":recall,
        "f1_score":f1
    }

def validate(args,model,val_loader,epoch,device):
    model.eval()
    all_predictions = []
    all_targets = []

    for sample_batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{args.epoch}'):
        inputs = sample_batch["text"]
        targets = sample_batch["label"]

        outputs,_ = model.predict(inputs, device)
        predictions = [0 if outputs[i][0] > outputs[i][1] else 1 for i in range(len(outputs))]

        all_predictions.extend(np.array(predictions))
        all_targets.extend(targets.cpu().numpy())

        batch_accuracy = accuracy_score(all_targets, all_predictions)
        print(f'Acc: {batch_accuracy:.4f}')

    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    auroc = roc_auc_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)

    return {
        "AUROC": auroc,
        "Acc": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def main(args):
    rank, world_size = setup_distributed()
    if args.device is None:
        device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    print('rank:', rank, 'world_size:', world_size, 'device:', device)

    print("Fetching data...")
    train_data_path = args.data_dir + args.train_data_file
    val_data_path = args.data_dir + args.val_data_file
    train_dataset=get_train_data(train_data_path)
    val_dataset = get_test_data(val_data_path)
    if "gpt" in args.pretrained_model:
        tokenizer=GPT2Tokenizer.from_pretrained(args.pretrained_model)
        pre_model=GPT2Model.from_pretrained(args.pretrained_model)
    else:
        if any(plm if plm in args.pretrained_model else False for plm in ["bart","electra","chinese"]):
            tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
            pre_model = AutoModel.from_pretrained(args.pretrained_model)
        else:
            tokenizer=RobertaTokenizer.from_pretrained(args.pretrained_model)
            pre_model = RobertaModel.from_pretrained(args.pretrained_model)
    pre_model.requires_grad_(False)
    weight_1=args.weight_1
    weight_sim = args.weight_2
    weight_ncd = args.weight_3

    model = CNetModel(pre_model,tokenizer)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=1,shuffle=True)
    criterion = CNetLoss()
    model.to(device)
    best_accuracy=0.0
    best_model_state=None
    for epoch in range(args.epoch):
        train_metric=train(args,model,optimizer,train_loader,criterion,weight_1,weight_sim,weight_ncd,epoch,device)
        train_auroc=train_metric["AUROC"]
        train_Acc=train_metric["Acc"]
        train_pre=train_metric["precision"]
        train_recall=train_metric["recall"]
        train_f1=train_metric["f1_score"]
        val_metric=validate(args,model,val_loader,epoch,device)
        val_auroc = val_metric["AUROC"]
        val_Acc = val_metric["Acc"]
        val_pre = val_metric["precision"]
        val_recall = val_metric["recall"]
        val_f1 = val_metric["f1_score"]
        print(
            f"train metric: AUROC:{train_auroc}  Acc:{train_Acc}  precision:{train_pre}  recall:{train_recall}  F1 score:{train_f1}")
        print(
            f"val metric: AUROC:{val_auroc}  Acc:{val_Acc}  precision:{val_pre}  recall:{val_recall}  F1 score:{val_f1}")
        if val_metric["Acc"]>best_accuracy:
            best_accuracy=val_metric["Acc"]
            best_model_state=model.state_dict()
    # Save the best model's state
    save_path=args.save_path
    if best_model_state is not None:
        torch.save(best_model_state, save_path)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='chatgpt/unfilter_full/')
    parser.add_argument('--train_data_file',type=str,default='train.csv')
    parser.add_argument('--val_data_file', type=str, default='val.csv')
    parser.add_argument('--optimizer',type=str,default='Adam')
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--epoch',type=int,default=1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--weight_1',type=float,default=1)
    parser.add_argument('--weight_2',type=float,default=0)
    parser.add_argument('--weight_3', type=float, default=1.2)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--l2_weight',type=float,default=0)
    parser.add_argument('--pretrained_model',type=str,default="facebookAI/roberta-base")
    parser.add_argument('--save_path',type=str,default='./model/model_tmp.pth')
    args, unparsed = parser.parse_known_args()
    main(args)