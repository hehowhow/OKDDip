import argparse
import logging
import os
import random
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import utils
import models.data_loader as data_loader
import models
import models.model_cifar as model_cifar
from tensorboardX import SummaryWriter


torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

model_names = sorted(name for name in model_cifar.__dict__
    if name.islower() and not name.startswith("__")
    and callable(model_cifar.__dict__[name]))

parser.add_argument('--model', metavar='ARCH', default='resnet32', type=str,
                    choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')    
parser.add_argument('--dataset', default='CIFAR100', type=str, help = 'Input the dataset name: default(CIFAR10)')
parser.add_argument('--num_epochs', default=300, type=int, help = 'Input the number of epoches: default(300)')
parser.add_argument('--batch_size', default=128, type=int, help = 'Input the batch size: default(128)')
parser.add_argument('--lr', default=0.1, type=float, help = 'Input the learning rate: default(0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--efficient', action='store_true', help = 'Decide whether or not to use efficient implementation: default(False)')
parser.add_argument('--wd', default=5e-4, type=float, help = 'Input the weight decay rate: default(5e-4)')
parser.add_argument('--dropout', default=0., type=float, help = 'Input the dropout rate: default(0.0)')
parser.add_argument('--resume', default='', type=str, help = 'Input the path of resume model: default('')')
parser.add_argument('--version', default='V0', type=str, help = 'Input the version of current model: default(V0)')
parser.add_argument('--num_workers', default=8, type=int, help = 'Input the number of works: default(8)')
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--data_root', default='./Data', type=str,
                    help='Dataset root directory: default(./Data)')

parser.add_argument('--num_branches', default=4, type=int, help = 'Input the number of branches: default(4)')
parser.add_argument('--loss', default='KL', type=str, help = 'Define the loss between student output and group output: default(KL_Loss)')
parser.add_argument('--temperature', default=3.0, type=float, help = 'Input the temperature: default(3.0)')
parser.add_argument('--alpha', default=1.0, type=float, help = 'Input the relative rate: default(1.0)')
parser.add_argument('--start_consistency', default=0., type=float, help = 'Input the start consistency rate: default(0.5)')
parser.add_argument('--length', default=80, type=float, help='length ratio: default(80)')
parser.add_argument('--MulStu', action='store_true', help = 'Decide whether or not to calculate multiStudent: default(False)')
parser.add_argument('--type', default='GL', type=str, help = 'Define the loss calculation strategy: default(GL)')
parser.add_argument('--lambda_ensemble', default=0.5, type=float, help = 'Weight for ensemble_logit in teacher signal fusion: default(0.5)')
parser.add_argument('--dissimilarity_metric', default='wasserstein1', type=str, 
                    choices=['wasserstein1', 'wasserstein2', 'euclidean', 'kl', 'cosine'],
                    help = 'Dissimilarity metric for adaptive weighting: wasserstein1(default), wasserstein2, euclidean, kl, cosine')
parser.add_argument('--tau', default=1.0, type=float, help = 'Temperature for softmax normalization in dissimilarity weighting: default(1.0)')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pdist = nn.PairwiseDistance(p=2)

def record_epoch_logits(model, sample_ids, ensemble_logits):
    history_model = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(history_model, 'record_epoch_logits'):
        history_model.record_epoch_logits(sample_ids, ensemble_logits)


def train(train_loader, model, optimizer, criterion, criterion_T, accuracy, args, consistency_weight):
    
    model.train()

    accTop1_avg = list(range(args.num_branches + 1))
    accTop5_avg = list(range(args.num_branches + 1))
    for i in range(args.num_branches + 1):
        accTop1_avg[i] = utils.RunningAverage()
        accTop5_avg[i] = utils.RunningAverage()
    loss_true_avg = utils.RunningAverage()
    loss_group_avg = utils.RunningAverage()
    loss_avg = utils.RunningAverage()    
    end = time.time()
    
    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (train_batch, labels_batch, sample_ids) in enumerate(train_loader):
            train_batch = train_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)
            sample_ids = sample_ids.cuda(non_blocking=True)
            
            model_output = model(train_batch, sample_ids=sample_ids)
            
            if len(model_output) == 4:
                output_batch, x_m, x_stu, ensemble_logit = model_output
                use_ensemble = True
                record_epoch_logits(model, sample_ids, ensemble_logit)
            else:
                output_batch, x_m, x_stu = model_output 
                use_ensemble = False
            
            mean_logit = torch.mean(output_batch, dim=2)
            
            loss_true = 0
            loss_group = 0    
            for i in range(args.num_branches - 1):
                loss_true += criterion(output_batch[:,:,i], labels_batch)
                
                if use_ensemble:
                    teacher_signal = args.lambda_ensemble * ensemble_logit + (1 - args.lambda_ensemble) * x_m[:,:,i]
                else:
                    teacher_signal = x_m[:,:,i]
                
                loss_group += criterion_T(output_batch[:,:,i], teacher_signal)
            
            if use_ensemble:
                student_teacher_signal = args.lambda_ensemble * ensemble_logit + (1 - args.lambda_ensemble) * mean_logit
            else:
                student_teacher_signal = mean_logit
            
            student_distill_loss = criterion_T(x_stu, student_teacher_signal)
            
            loss = loss_true + criterion(x_stu, labels_batch) + args.alpha * consistency_weight * (loss_group + student_distill_loss)
        
            loss_true_avg.update(loss_true.item())
            loss_group_avg.update(loss_group.item())
            loss_avg.update(loss.item())
            
            for i in range(args.num_branches - 1):
                metrics = accuracy(output_batch[:,:,i], labels_batch, topk=(1,5))
                accTop1_avg[i].update(metrics[0].item())
                accTop5_avg[i].update(metrics[1].item())
                
            metrics = accuracy(x_stu, labels_batch, topk=(1,5))
            accTop1_avg[args.num_branches - 1].update(metrics[0].item())
            accTop5_avg[args.num_branches - 1].update(metrics[1].item())
        
            e_metrics = accuracy(torch.mean(output_batch, dim=2), labels_batch, topk=(1,5))
            accTop1_avg[args.num_branches].update(e_metrics[0].item())
            accTop5_avg[args.num_branches].update(e_metrics[1].item())
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
            
            t.update()
            
    mean_train_accTop1 = 0
    mean_train_accTop5 = 0
    for i in range(args.num_branches - 1):
        mean_train_accTop1 += accTop1_avg[i].value()
        mean_train_accTop5 += accTop5_avg[i].value()
    mean_train_accTop1 /= (args.num_branches-1)
    mean_train_accTop5 /= (args.num_branches-1)
    
    train_metrics = {'train_loss': loss_avg.value(),
                     'train_true_loss': loss_true_avg.value(),
                     'train_group_loss': loss_group_avg.value(),
                     'mean_train_accTop1': mean_train_accTop1,
                     'mean_train_accTop5': mean_train_accTop1,
                     'stu_train_accTop1': accTop1_avg[args.num_branches - 1].value(),
                     'stu_train_accTop5': accTop5_avg[args.num_branches - 1].value(),
                     'train_accTop1': accTop1_avg[args.num_branches].value(),
                     'train_accTop5': accTop5_avg[args.num_branches].value(),
                     'time': time.time() - end}
                     
    for i in range(args.num_branches - 1):
        train_metrics.update({'stu'+str(i)+'train_accTop1' : accTop1_avg[i].value()})
        train_metrics.update({'stu'+str(i)+'train_accTop5' : accTop5_avg[i].value()})
        
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    logging.info("- Train metrics: " + metrics_string)
    return train_metrics

    
def evaluate(test_loader, model, criterion, criterion_T, accuracy, args, consistency_weight):
    model.eval()
    
    accTop1_avg = list(range(args.num_branches + 1))
    accTop5_avg = list(range(args.num_branches + 1))
    for i in range(args.num_branches + 1):
        accTop1_avg[i] = utils.RunningAverage()
        accTop5_avg[i] = utils.RunningAverage()
    
    loss_true_avg = utils.RunningAverage()
    loss_group_avg = utils.RunningAverage()
    loss_avg = utils.RunningAverage()
    dist_avg = utils.RunningAverage()
    end = time.time()
    
    with torch.no_grad():
        for batch_idx, (test_batch, labels_batch, _) in enumerate(test_loader):
            test_batch = test_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)
            
            loss_true = 0
            loss_group = 0
    
            model_output = model(test_batch, sample_ids=None)
            
            if len(model_output) == 4:
                output_batch, x_m, x_stu, ensemble_logit = model_output
                use_ensemble = True
            else:
                output_batch, x_m, x_stu = model_output
                use_ensemble = False
            
            mean_logit = torch.mean(output_batch, dim=2)
            
            for i in range(args.num_branches - 1):
                loss_true += criterion(output_batch[:,:,i], labels_batch)
                
                if use_ensemble:
                    teacher_signal = args.lambda_ensemble * ensemble_logit + (1 - args.lambda_ensemble) * x_m[:,:,i]
                else:
                    teacher_signal = x_m[:,:,i]
                
                loss_group += criterion_T(output_batch[:,:,i], teacher_signal)
            
            if use_ensemble:
                student_teacher_signal = args.lambda_ensemble * ensemble_logit + (1 - args.lambda_ensemble) * mean_logit
            else:
                student_teacher_signal = mean_logit
            
            student_distill_loss = criterion_T(x_stu, student_teacher_signal)
            
            loss = loss_true + criterion(x_stu, labels_batch) + args.alpha * consistency_weight * (loss_group + student_distill_loss)
    
            loss_true_avg.update(loss_true.item())
            loss_group_avg.update(loss_group.item())
            loss_avg.update(loss.item())
            
            for i in range(args.num_branches - 1):
                metrics = accuracy(output_batch[:,:,i], labels_batch, topk=(1,5))
                accTop1_avg[i].update(metrics[0].item())
                accTop5_avg[i].update(metrics[1].item())
                                
            metrics = accuracy(x_stu, labels_batch, topk=(1,5))
            accTop1_avg[args.num_branches - 1].update(metrics[0].item())
            accTop5_avg[args.num_branches - 1].update(metrics[1].item())
                
            e_metrics = accuracy(torch.mean(output_batch, dim=2), labels_batch, topk=(1,5))
            accTop1_avg[args.num_branches].update(e_metrics[0].item())
            accTop5_avg[args.num_branches].update(e_metrics[1].item()) 
            
            len_kk = output_batch.size(0)
            output_batch = F.softmax(output_batch, dim=1)    
            for kk in range(len_kk):
                ret = output_batch[kk,:,:]
                ret = ret.t()
                sim = 0
                for j in range(args.num_branches-1):
                    for k in range(j+1, args.num_branches-1):
                        sim += pdist(ret[j:j+1,:],ret[k:k+1,:])    
                sim = sim / 3
                dist_avg.update(sim.item())

    mean_test_accTop1 = 0
    mean_test_accTop5 = 0
    for i in range(args.num_branches - 1):
        mean_test_accTop1 += accTop1_avg[i].value()
        mean_test_accTop5 += accTop5_avg[i].value()
    mean_test_accTop1 /= (args.num_branches - 1)
    mean_test_accTop5 /= (args.num_branches - 1)
        
    test_metrics = { 'test_loss': loss_avg.value(),
                     'test_true_loss': loss_true_avg.value(),
                     'test_group_loss': loss_group_avg.value(),
                     'mean_test_accTop1': mean_test_accTop1,
                     'mean_test_accTop5': mean_test_accTop5,
                     'test_accTop1': accTop1_avg[args.num_branches].value(),
                     'test_accTop5': accTop5_avg[args.num_branches].value(),
                     'stu_test_accTop1': accTop1_avg[args.num_branches - 1].value(),
                     'stu_test_accTop5': accTop5_avg[args.num_branches - 1].value(),
                     'dist': dist_avg.value(),
                     'time': time.time() - end}
    for i in range(args.num_branches - 1):
        test_metrics.update({'stu'+str(i)+'test_accTop1' : accTop1_avg[i].value()})
        test_metrics.update({'stu'+str(i)+'test_accTop5' : accTop5_avg[i].value()})
    
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test metrics: " + metrics_string)
    return test_metrics

def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, criterion_T, accuracy, model_dir, args, timestamp):
    
    start_epoch = 0
    best_acc = 0.
        
    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    
    writer = SummaryWriter(log_dir = model_dir)
    writerB = SummaryWriter(log_dir = os.path.join(model_dir, 'B'))
    
    choose_E = False
    
    result_train_metrics = list(range(args.num_epochs))
    result_test_metrics = list(range(args.num_epochs))
    
    if args.resume:
        logging.info('Resuming from checkpoint..')
        resumePath = os.path.join(args.resume, 'last.pth')
        assert os.path.isfile(resumePath), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath)        
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        start_epoch = checkpoint['epoch']
        scheduler.step(start_epoch - 1)
        
        if choose_E:
            best_acc = checkpoint['test_accTop1']
        else:
            best_acc = checkpoint['stu_test_accTop1']
        result_train_metrics = torch.load(os.path.join(args.resume, 'train_metrics'))
        result_test_metrics = torch.load(os.path.join(args.resume, 'test_metrics'))
        
    for epoch in range(start_epoch, args.num_epochs):
        
        scheduler.step()
     
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))
        
        consistency_epoch = args.start_consistency * args.num_epochs 
        if epoch < consistency_epoch:
            consistency_weight = 1
        else:
            consistency_weight = get_current_consistency_weight(epoch - consistency_epoch, args.length)
        
        train_metrics = train(train_loader, model, optimizer, criterion, criterion_T, accuracy, args, consistency_weight)
		
        writer.add_scalar('Train/Loss', train_metrics['train_loss'], epoch+1)
        writer.add_scalar('Train/Loss_True', train_metrics['train_true_loss'], epoch+1)
        writer.add_scalar('Train/Loss_Group', train_metrics['train_group_loss'], epoch+1)
        writer.add_scalar('Train/AccTop1', train_metrics['train_accTop1'], epoch+1)
        writerB.add_scalar('Train/AccTop1', train_metrics['stu_train_accTop1'], epoch+1)
        writerB.add_scalar('Train/AccTop1_B0', train_metrics['stu0train_accTop1'], epoch+1)
        writerB.add_scalar('Train/AccTop1_B1', train_metrics['stu1train_accTop1'], epoch+1)
        writerB.add_scalar('Train/AccTop1_B2', train_metrics['stu2train_accTop1'], epoch+1)
    
        test_metrics = evaluate(test_loader, model, criterion, criterion_T, accuracy, args, consistency_weight) 
        
        if hasattr(model, 'update_epoch_history'):
            if isinstance(model, nn.DataParallel):
                model.module.update_epoch_history()
            else:
                model.update_epoch_history()
            logging.info(f"- Updated epoch history. Epoch count: {model.epoch_count if not isinstance(model, nn.DataParallel) else model.module.epoch_count}")
        
        if choose_E:
            test_acc = test_metrics['test_accTop1']
        else:
            test_acc = test_metrics['stu_test_accTop1']
            
        writer.add_scalar('Test/Loss', test_metrics['test_loss'], epoch+1)
        writer.add_scalar('Test/Loss_True', test_metrics['test_true_loss'], epoch+1)
        writer.add_scalar('Test/Loss_Group', test_metrics['test_group_loss'], epoch+1)
        writer.add_scalar('Test/AccTop1', test_metrics['test_accTop1'], epoch+1)
        writerB.add_scalar('Test/AccTop1', test_metrics['stu_test_accTop1'], epoch+1)
        writerB.add_scalar('Test/AccTop1_B0', test_metrics['stu0test_accTop1'], epoch+1)
        writerB.add_scalar('Test/AccTop1_B1', test_metrics['stu1test_accTop1'], epoch+1)
        writerB.add_scalar('Test/AccTop1_B2', test_metrics['stu2test_accTop1'], epoch+1)
        
        result_train_metrics[epoch] = train_metrics
        result_test_metrics[epoch] = test_metrics
        
        torch.save(result_train_metrics, os.path.join(model_dir, 'train_metrics'))
        torch.save(result_test_metrics, os.path.join(model_dir, 'test_metrics'))

        last_path = os.path.join(model_dir, 'last.pth')
        torch.save({    'state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'optim_dict': optimizer.state_dict(),
                        'test_accTop1': test_metrics['test_accTop1'],
                        'mean_test_accTop1': test_metrics['mean_test_accTop1'],
                        'stu_test_accTop1': test_metrics['stu_test_accTop1']}, last_path)
        is_best = test_acc >= best_acc
        if is_best:
            logging.info("- Found better accuracy")            
            best_acc = test_acc
            test_metrics['epoch'] = epoch + 1
            best_metrics_filename = f"test_best_metrics_ensem_{args.gpu_id}_{args.lambda_ensemble}_tau1.5_{args.model}_{args.dataset}_{timestamp}.json"
            utils.save_dict_to_json(test_metrics, os.path.join(model_dir, best_metrics_filename))
        
            shutil.copyfile(last_path, os.path.join(model_dir, 'best.pth'))
    writer.close()

def get_current_consistency_weight(current, rampup_length = args.length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

if __name__ == '__main__':

    begin_time = time.time()
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.MulStu:
        model_dir= os.path.join('.', args.dataset, str(args.num_epochs), args.type, args.model + 'M' + str(args.num_branches) + 'T' + str(args.temperature) + 'S' + str(args.loss) + args.version)
    else:
        model_dir= os.path.join('.', args.dataset, str(args.num_epochs), args.type, args.model + 'B' + str(args.num_branches) + 'T' + str(args.temperature) + 'S' + str(args.loss) + args.version)
    
    if not os.path.exists(model_dir):
        print("Directory does not exist! Making directory {}".format(model_dir))
        os.makedirs(model_dir)
    
    log_filename = f'train_{args.model}_{args.dataset}_{args.gpu_id}_{args.lambda_ensemble}_{timestamp}.log'
    utils.set_logger(os.path.join(model_dir, log_filename))

    logging.info("Loading the datasets...")
    
    if args.dataset == 'CIFAR10':
        num_classes = 10
        model_folder = "model_cifar"
    elif args.dataset == 'CIFAR100':
        num_classes = 100
        model_folder = "model_cifar"
    elif args.dataset == 'imagenet':
        num_classes = 1000
        model_folder = "model_imagenet"
    root = args.data_root
    
    train_loader, test_loader = data_loader.dataloader(
        data_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root=root,
        return_indices=True,
    )
    logging.info("- Done.")
    
    model_fd = getattr(models, model_folder)
    if args.MulStu:
        model_cfg = getattr(model_fd, 'MultiNet')
        model = getattr(model_cfg, 'StuNet')(model = args.model, num_branches = args.num_branches, num_classes = num_classes, input_channel=utils.lookup(args.model), dropout = args.dropout)
    elif args.type == 'DML':
        model_cfg = getattr(model_fd, 'DML')
        model = getattr(model_cfg, 'MutualNet')(model = args.model, num_branches = args.num_branches, num_classes = num_classes)
    else:
        if "resnet" in args.model:
            model_cfg = getattr(model_fd, 'resnet_GL')
            model = getattr(model_cfg, args.model)(num_classes = num_classes, num_branches = args.num_branches, 
                                                   input_channel=utils.lookup(args.model), 
                                                   dissimilarity_metric=args.dissimilarity_metric,
                                                   tau=args.tau)
        elif "vgg" in args.model:
            model_cfg = getattr(model_fd, 'vgg_GL')
            model = getattr(model_cfg, args.model)(num_classes = num_classes, num_branches = args.num_branches,
                                                   dissimilarity_metric=args.dissimilarity_metric,
                                                   tau=args.tau)
        elif "densenet" in args.model:
            model_cfg = getattr(model_fd, 'densenet_GL')
            model = getattr(model_cfg, args.model)(num_classes = num_classes, num_branches = args.num_branches,
                                                   dissimilarity_metric=args.dissimilarity_metric,
                                                   tau=args.tau)
        
        
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0,1,2,3]).to(device)
    else:
        model = model.to(device)
    
    num_params = (sum(p.numel() for p in model.parameters())/1000000.0)
    logging.info('Total params: %.2fM' % num_params)
    
    criterion = nn.CrossEntropyLoss()
    if args.loss == "KL":
        criterion_T = utils.KL_Loss(args.temperature).to(device)
    elif args.loss == "CE":
        criterion_T = utils.CE_Loss(args.temperature).to(device)
    
    accuracy = utils.accuracy
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay = args.wd)    
    
    logging.info("Starting training for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, criterion_T, accuracy, model_dir, args, timestamp)
    
    logging.info('Total time: {:.2f} hours'.format((time.time() - begin_time)/3600.0))
    state['Total params'] = num_params
    params_json_path = os.path.join(model_dir, "parameters.json")
    utils.save_dict_to_json(state, params_json_path)
