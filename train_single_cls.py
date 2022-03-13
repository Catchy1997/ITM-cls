# -*- coding:utf-8 -*-
import os, time, sys
import shutil

import torch
print("Torch version:", torch.__version__)
# from tensorboardX import SummaryWriter

from utils._common import init_logger, logger
from utils._config import entertain as cfg
from utils._config import get_argparse
from utils._mydataset import my_loader, gene_dataloder
from utils import AverageMeter
from utils._metrics import accuracy, adjust_learning_rate
from utils._model_builder import model_builder, save_checkpoint
from utils._transforms import transform_dict
from utils.progressbar import ProgressBar

from models.dnn import Net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args, train_root, model, transform_dic):
    # load the data
    train_loader = gene_dataloder(args, data_root=train_root, loader=my_loader)
    
    # Prepare criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_loader))
    logger.info("  Num Epochs = %d", cfg['end_epoch'])
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.batch_size)

    global_step = 0
    
    tr_loss, min_loss = AverageMeter(), float('inf')
    model.zero_grad()
    for epoch in range(cfg['end_epoch']):
        pbar = ProgressBar(n_total=len(train_loader), desc='Training')
        for step, batch in enumerate(train_loader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            img, text, labels = torch.autograd.Variable(batch[0]), torch.autograd.Variable(batch[1]), torch.autograd.Variable(batch[2]) # 转换数据格式
            # optimizer.zero_grad() # 梯度置零，因为反向传播过程中梯度会累加上一次循环的梯度
            outputs, _ = model(img, text)
            # prec, _ = accuracy(outputs.data, labels, topk=(1,1))
            loss = criterion(outputs.view(args.batch_size,-1).contiguous(), labels) # 计算损失值
            loss.backward() # loss反向传播 计算反向梯度      
            pbar(step, {'loss': loss.item(), 'lr': args.lr})            
            tr_loss.update(loss.item(), n=1) # loss累加

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Save model checkpoint         
                is_best = loss.item() < min_loss
                min_loss = min(loss.item(), min_loss)
                # save_checkpoint(args, model, global_step, is_best)
                # args.lr = adjust_learning_rate(args, optimizer, global_step, cfg['epoch_step'], len(train_loader), step)
    
    # Evaluation
    print("\n")
    eval_loss = evaluate(args, model, transform_dic)
    print("\n")
    logger.info(" Eval loss = {:.4f}".format(eval_loss))

    if 'cuda' in str(device):
        torch.cuda.empty_cache()
    
    return global_step, tr_loss.avg


def evaluate(args, model, transform_dic, val_root=cfg["Valroot"]):
    eval_output_dir = args.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    val_loader = gene_dataloder(args, data_root=val_root, loader=my_loader)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(val_loader))
    logger.info("  Batch size = %d", args.batch_size)
    
    eval_loss, eval_prec = AverageMeter(), AverageMeter()
    pbar = ProgressBar(n_total=len(val_loader), desc="Evaluating")
    for step, batch in enumerate(val_loader):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            img, text, labels = torch.autograd.Variable(batch[0]), torch.autograd.Variable(batch[1]), torch.autograd.Variable(batch[2]) # 转换数据格式
            outputs, _ = model(img, text)
            outputs = outputs.view(args.batch_size, -1).contiguous()
            tmp_eval_loss = criterion(outputs, labels) # 计算损失值
            eval_loss.update(tmp_eval_loss.item(), n=1)
            
            prec, _ = accuracy(outputs.data, labels, topk=(1, 1))
            eval_prec.update(prec, n=1)
        pbar(step, {'loss': eval_loss.avg, 'precision': eval_prec.avg})
    
    return eval_loss.avg


def main():
    args = get_argparse()
    # pretrained_model = cfg["pretrained_dict"][args.model_name]

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + f'/{time_}.log')

    # Setup CUDA, GPU & distributed training
    if args.cuda and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    logger.warning("Process device: %s", device)

    # Load model
    # net = model_builder(args.model_name, pretrained=True, weight_path=pretrained_model, num_classes=cfg['num_classes'])
    net = Net(in_shape=1024, num_classes=cfg['num_classes'])
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net)
    if args.cuda:
        net.cuda()
        torch.backends.cudnn.benchmark = True
    logger.info("Training/evaluation parameters %s", args)

    # Training
    global_step, tr_loss = train(args, cfg["Trainroot"], net, transform_dict)
    logger.info("global_step = {}, average loss = {:.4f}".format(global_step, tr_loss))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()