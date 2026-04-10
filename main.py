import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml
from data_utils import genSpoof_list, Dataset_ASVspoof2019_train, Dataset_ASVspoof2021_eval
from model import RawNet
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed

def evaluate_accuracy(dev_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for batch_x, batch_y in dev_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
    return 100 * (num_correct / num_total)

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    model.eval()
    for batch_x, utt_id in data_loader:
        fname_list = []
        score_list = []
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write(f'{f} {cm}\n')
    print(f'Scores saved to {save_path}')

def train_epoch(train_loader, model, lr, optim, device):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    model.train()
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in train_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RawNet2 ASVspoof Training')
    parser.add_argument('--database_path', type=str, default='./data/LA/', help='Base data directory')
    parser.add_argument('--protocols_path', type=str, default='./data/LA/', help='Protocols path')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--track', type=str, default='LA', choices=['LA'])
    parser.add_argument('--eval_output', type=str, default=None)
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument('--is_eval', action='store_true', default=False)
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', default=True)
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', default=False)

    dir_yaml = 'model_config_RawNet.yaml'
    with open(dir_yaml, 'r') as f_yaml:
        parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)

    os.makedirs('models', exist_ok=True)
    args = parser.parse_args()
    set_random_seed(args.seed, args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device)

    model = RawNet(parser1['model'], device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model_tag = f'model_{args.track}_{args.loss}_{args.num_epochs}_{args.batch_size}_{args.lr}'
    if args.comment:
        model_tag += f'_{args.comment}'
    model_save_path = os.path.join('models', model_tag)
    os.makedirs(model_save_path, exist_ok=True)

    if args.eval:
        eval_file = genSpoof_list(os.path.join(args.protocols_path, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2021.LA.cm.eval.trl.txt'), is_eval=True)
        eval_set = Dataset_ASVspoof2021_eval(list_IDs=eval_file, base_dir=os.path.join(args.database_path, 'ASVspoof2021_LA_eval/'))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)

    d_label_trn, file_train = genSpoof_list(os.path.join(args.protocols_path, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.train.trn.txt'), is_train=True)
    train_set = Dataset_ASVspoof2019_train(file_train, d_label_trn, os.path.join(args.database_path, 'ASVspoof2019_LA_train/'))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    d_label_dev, file_dev = genSpoof_list(os.path.join(args.protocols_path, 'ASVspoof2019_LA_cm_protocols', 'ASVspoof2019.LA.cm.dev.trl.txt'))
    dev_set = Dataset_ASVspoof2019_train(file_dev, d_label_dev, os.path.join(args.database_path, 'ASVspoof2019_LA_dev/'))
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False)

    writer = SummaryWriter(f'logs/{model_tag}')
    best_acc = 99
    for epoch in range(args.num_epochs):
        running_loss, train_acc = train_epoch(train_loader, model, args.lr, optimizer, device)
        val_acc = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print(f'\nEpoch {epoch} | Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            print('Saving best model at epoch', epoch)
        best_acc = max(val_acc, best_acc)
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch_{epoch}.pth'))