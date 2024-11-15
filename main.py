import warnings
warnings.filterwarnings("ignore")
import argparse
import json
import logging
import os
from time import time
# from sklearn.metrics import f1_score
import dgl
import data_process
import torch
import torch.nn
import torch.nn.functional as F
from dgl.data import LegacyTUDataset
from dgl.dataloading import GraphDataLoader
from model import HGPSLModel
from torch.utils.data import random_split
from utils import get_stats

# 1. We can't replicate the excellent performance as the paper author said in the issues of the official github of the paper
# 2. Other people trying to replicate the paper also have the same issue and having a large accuracy gap between the paper reported and the actual result
# TODO: prove the accuracy varies when changing the random seed
# TODO: add a F1-score metrics for the test set
# TODO: change the main function to allow us compare the original implementation with the modified one

def parse_args():
    parser = argparse.ArgumentParser(description="HGP-SL-DGL")
    parser.add_argument(
        "--random_seed", type=int, default=777, help="random seed for random split of test,val,train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size"
    )
    parser.add_argument(
        "--sample", type=str, default="true", help="use sample method"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-3, help="weight decay"
    )
    parser.add_argument(
        "--pool_ratio", type=float, default=0.7, help="pooling ratio"
    )
    parser.add_argument("--hid_dim", type=int, default=128, help="hidden size")
    parser.add_argument(
        "--conv_layers", type=int, default=2, help="number of conv layers"
    )
    parser.add_argument(
        "--pool_layers", type=float, default=1, help="# pooling layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout ratio"
    )
    parser.add_argument(
        "--lamb", type=float, default=1.0, help="trade-off parameter"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="max number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=100, help="patience for early stopping"
    )
    parser.add_argument(
        "--device", type=int, default=-1, help="device id, -1 for cpu"
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=10,
        help="print trainlog every k epochs, -1 for silent training",
    )
    parser.add_argument(
        "--num_trials", type=int, default=3, help="number of trials"
    )
    parser.add_argument("--output_path", type=str, default="./output")
    parser.add_argument("--model_path", type=str, default="./models")
    
    args = parser.parse_args()

    # device
    args.device = "cpu" if args.device == -1 else "cuda:{}".format(args.device)
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available, use CPU for training.")
        args.device = "cpu"

    # print every
    if args.print_every == -1:
        args.print_every = args.epochs + 1

    # bool args
    if args.sample.lower() == "true":
        args.sample = True
    else:
        args.sample = False

    # # paths
    # if not os.path.exists(args.output_path):
    #     os.makedirs(args.output_path)
    # name = (
    #     "Hidden={}_Pool={}_WeightDecay={}_Lr={}_Sample={}.log".format(
    #         args.hid_dim,
    #         args.pool_ratio,
    #         args.weight_decay,
    #         args.lr,
    #         args.sample,
    #     )
    # )
    # args.output_path = os.path.join(args.output_path, name)
        # paths
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    name = (
        "#CONV={}_#POOL={}_Hidden={}_PoolR={}_RANDS={}.log".format(
            args.conv_layers,
            args.pool_layers,
            args.hid_dim,
            args.pool_ratio,
            args.random_seed,
        )
    )
    args.output_path = os.path.join(args.output_path, name)
    
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    modelname = (
        "#CONV={}_#POOL={}_Hidden={}_PoolR={}_RANDS={}.pth".format(
            args.conv_layers,
            args.pool_layers,
            args.hid_dim,
            args.pool_ratio,
            args.random_seed,
        )
    )
    args.model_path = os.path.join(args.model_path, modelname)

    return args


def train(model: torch.nn.Module, optimizer, trainloader, device):
    model.train()
    total_loss = 0.0
    num_batches = len(trainloader)
    for batch in trainloader:
        optimizer.zero_grad()
        batch_graphs, batch_labels = batch
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs, batch_graphs.ndata["feat"])
        loss = F.nll_loss(out, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / num_batches


@torch.no_grad()
def test(model: torch.nn.Module, loader, device):
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)
        out = model(batch_graphs, batch_graphs.ndata["feat"])
        pred = out.argmax(dim=1)
        
        # all_preds.extend(pred.cpu().numpy())
        # all_labels.extend(batch_labels.cpu().numpy())
        
        loss += F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()
        
    # return correct / num_graphs, loss / num_graphs , f1_score(all_labels, all_preds, average="macro")
    return correct / num_graphs, loss / num_graphs

def main(args):
    
    # Step 1: Prepare graph data and retrieve train/validation/test index ============================= #
    dataset = LegacyTUDataset("PROTEINS")

    # add self loop. We add self loop for each graph here since the function "add_self_loop" does not
    # support batch graph.
    for i in range(len(dataset)):
        dataset.graph_lists[i] = dgl.add_self_loop(dataset.graph_lists[i])

    num_training = int(len(dataset) * 0.8)
    num_val = int(len(dataset) * 0.1)
    num_test = len(dataset) - num_val - num_training
    
    
    train_set, val_set, test_set = random_split(
        dataset, [num_training, num_val, num_test], generator=torch.Generator().manual_seed(args.random_seed)
    )
    short_proteins_test = data_process.select_subset_sizecriteria(dataset,test_set,"short")
    long_proteins_test = data_process.select_subset_sizecriteria(dataset,test_set,"long")

    train_loader = GraphDataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=6
    )
    val_loader = GraphDataLoader(
        val_set, batch_size=args.batch_size, num_workers=2
    )
    test_loader = GraphDataLoader(
        test_set, batch_size=args.batch_size, num_workers=2
    )
    
    test_loader_shortsize = GraphDataLoader(
        short_proteins_test, batch_size=args.batch_size, num_workers=2
    )

    test_loader_longsize = GraphDataLoader(
        long_proteins_test, batch_size=args.batch_size, num_workers=2
    )
    
    device = torch.device(args.device)

    # Step 2: Create model =================================================================== #
    num_feature, num_classes, _ = dataset.statistics()

    model = HGPSLModel(
        in_feat=num_feature,
        out_feat=num_classes,
        hid_feat=args.hid_dim,
        conv_layers=args.conv_layers,
        pool_layers=args.pool_layers,
        dropout=args.dropout,
        pool_ratio=args.pool_ratio,
        lamb=args.lamb,
        sample=args.sample,
    ).to(device)
    args.num_feature = int(num_feature)
    args.num_classes = int(num_classes)
    # Define the path to save the best model
    best_model_path = args.model_path
    
    # Step 3: Create training components ===================================================== #
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Step 4: training epoches =============================================================== #
    bad_cound = 0
    best_val_loss = float("inf")
    final_test_acc = 0.0
    final_test_acc_short = 0.0
    final_test_acc_long = 0.0
    # final_test_f1 = 0.0
    # final_test_f1_short = 0.0
    # final_test_f1_long = 0.0
    train_times = []
    for e in range(args.epochs):
        s_time = time()
        train_loss = train(model, optimizer, train_loader, device)
        train_times.append(time() - s_time)
        val_acc, val_loss= test(model, val_loader, device)
        test_acc, _ = test(model, test_loader, device)
        test_acc_short, _ = test(model, test_loader_shortsize, device)
        test_acc_long, _ = test(model, test_loader_longsize, device)        
        val_acc, val_loss= test(model, val_loader, device)
        # test_acc, _ , f1= test(model, test_loader, device)
        # test_acc_short, _ , f1_short = test(model, test_loader_shortsize, device)
        # test_acc_long, _ , f1_long = test(model, test_loader_longsize, device)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            final_test_acc = test_acc
            # final_test_f1 = f1
            # final_test_f1_long = f1_long
            # final_test_f1_short = f1_short
            final_test_acc_short = test_acc_short
            final_test_acc_long = test_acc_long
            
            best_epoch = e + 1
            # Save the best model
            torch.save(model.state_dict(), best_model_path)

        else:
            bad_cound += 1
        if bad_cound >= args.patience:
            break

        if (e + 1) % args.print_every == 0:
            log_format = (
                "Epoch {}: loss={:.4f}, val_acc={:.4f}, final_test_acc={:.4f}, final test acc for short={:.4f}, final test acc for long={:.4f} "
            )
            print(log_format.format(e + 1, train_loss, val_acc, final_test_acc, final_test_acc_short, final_test_acc_long))
    print(
        "Best Epoch {}, final test acc ={:.4f}, final test acc for short={:.4f}, final test acc for long={:.4f}".format(
            best_epoch, final_test_acc, final_test_acc_short, final_test_acc_long
        )
    )
    return best_val_loss,final_test_acc, final_test_acc_short, final_test_acc_long, sum(train_times) / len(train_times)


if __name__ == "__main__":
    args = parse_args()
    res = []
    res_short = []
    res_long = []
    # f1s =[]
    # f1s_short = []
    # f1s_long = []
    train_times = []
    for i in range(args.num_trials):
        print("Trial {}/{}".format(i + 1, args.num_trials))
        # acc,f1,acc_short,f1_short, acc_long,f1_long,train_time = main(args)
        _,acc,acc_short,acc_long,train_time = main(args)
        res.append(acc)
        res_short.append(acc_short)
        res_long.append(acc_long)
        # f1s.append(f1)
        # f1s_short.append(f1_short)
        # f1s_long.append(f1_long)
        train_times.append(train_time)

    mean, err_bd = get_stats(res, conf_interval=False)
    mean_short,err_bd_short = get_stats(res_short, conf_interval=False)
    mean_long,err_bd_long = get_stats(res_long, conf_interval=False)
    
    # mean_f1,err_bd_f1 = get_stats(f1s,conf_interval=False)
    # mean_short_f1,err_bd_short_f1 = get_stats(f1s_short, conf_interval=False)
    # mean_long_f1,err_bd_long_f1 = get_stats(f1s_long, conf_interval=False)
    # print("mean acc: {:.4f}, error bound: {:.4f}".format(mean, err_bd))
    # print("for short proteins, mean acc: {:.4f}, error bound: {:.4f}".format(mean_short, err_bd_short))
    # print("for long proteins, mean acc: {:.4f}, error bound: {:.4f}".format(mean_long, err_bd_long))
    
    out_dict = {
        "hyper-parameters": vars(args),
        "accuracy": "{:.4f}(+-{:.4f})".format(mean, err_bd),
        "accuracy_short" : "{:.4f}(+-{:.4f})".format(mean_short, err_bd_short),
        "accuracy_long" : "{:.4f}(+-{:.4f})".format(mean_long, err_bd_long),
    
        # "f1":"{:.4f}(+-{:.4f}) ".format(mean_f1, err_bd_f1),
        # "f1_short":"{:.4f}(+-{:.4f}) ".format(mean_short_f1, err_bd_short_f1),
        # "f1_long":"{:.4f}(+-{:.4f}) ".format(mean_long_f1, err_bd_long_f1),
        
        "train_time": "{:.4f}".format(sum(train_times) / len(train_times)),
    }

    with open(args.output_path, "w") as f:
        json.dump(out_dict, f, sort_keys=True, indent=4)
