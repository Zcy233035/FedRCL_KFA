import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random

import datetime
#from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from backpack import extend, backpack
from backpack.extensions import KFAC, KFRA, KFLR
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MLP', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='communication strategy: fedavg/fedprox')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')
    parser.add_argument('--coor', type=float, default=0.0001, help='Parameter controlling the wight of FIM')
    args = parser.parse_args()
    return args

def init_nets_extend(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if args.dataset == "generated":
            net = PerceptronModel()
        elif args.model == "test":
            net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16,8]
            elif args.dataset == 'fmnist':
                input_size = 784
                output_size = 10
                hidden_sizes = [254,64]
            net = SeFcNet(input_size, hidden_sizes, output_size, dropout_p)
            net.layers = extend(net.layers)
        elif args.model == "mlpfull":
            input_size = 784
            output_size = 10
            hidden_sizes = [10]
            net = SeFcNet(input_size, hidden_sizes, output_size, dropout_p)
            net.layers = extend(net.layers)
        elif args.model == "vgg":
            net = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
            net.layers = extend(net.layers)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        elif args.model == "resnet18":
            net = ResNet18_cifar10()
        elif args.model == "resnet50":
            net = ResNet50_cifar10()
        elif args.model == "vgg16":
            net = vgg16()
        else:
            print("not supported yet")
            exit(1)

        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def init_nets(net_configs, dropout_p, n_parties, args):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):
        if args.dataset == "generated":
            net = PerceptronModel()
        elif args.model == "test":
            net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))
        elif args.model == "mlp":
            if args.dataset == 'covtype':
                input_size = 54
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'a9a':
                input_size = 123
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'rcv1':
                input_size = 47236
                output_size = 2
                hidden_sizes = [32,16,8]
            elif args.dataset == 'SUSY':
                input_size = 18
                output_size = 2
                hidden_sizes = [16,8]
            elif args.dataset == 'fmnist':
                input_size = 784
                output_size = 10
                hidden_sizes = [254,64]
            net = SeFcNet(input_size, hidden_sizes, output_size, dropout_p)
        elif args.model == "vgg":
            net = vgg11()
        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
        elif args.model == "vgg-9":
            if args.dataset in ("mnist", 'femnist'):
                net = ModerateCNNMNIST()
            elif args.dataset in ("cifar10", "cinic10", "svhn"):
                # print("in moderate cnn")
                net = ModerateCNN()
            elif args.dataset == 'celeba':
                net = ModerateCNN(output_dim=2)
        elif args.model == "resnet":
            net = ResNet50_cifar10()
        elif args.model == "vgg16":
            net = vgg16()
        else:
            print("not supported yet")
            exit(1)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type

def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)

def local_train_net_riem(train_net_func, nets, selected, f_nets, args, net_dataidx_map, coordinator, test_dl = None, device="cpu", avg_coor=0.0):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs
        if avg_coor>0:
            trainacc, testacc = train_net_func(net_id, net, f_nets[net_id], len(dataidxs), train_dl_local, test_dl, n_epoch, args.lr, coordinator, args.optimizer, device=device, avg_coor=avg_coor)
        else:
            trainacc, testacc = train_net_func(net_id, net, f_nets[net_id], len(dataidxs), train_dl_local, test_dl, n_epoch, args.lr, coordinator, args.optimizer, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

def local_train_net_blockoffline(nets, selected, f_nets, args, net_dataidx_map, test_dl = None, device="cpu"):
    return local_train_net_riem(train_net_blockoffline, nets, selected, f_nets, args, net_dataidx_map, args.coor, test_dl, device)

def train_net_blockoffline(net_id, net, f_local, len_of_data, train_dataloader, test_dataloader, epochs, lr, coordinator, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    criterion = extend(nn.CrossEntropyLoss()).to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()

    for epoch in range(epochs+1):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()
                out = net(x)
                loss = criterion(out, target)

                if epoch<epochs:
                    loss.backward()
                    optimizer.step()
                    cnt += 1
                    epoch_loss_collector.append(loss.item())
                else:
                    with backpack(KFRA()): # using KFRA to evlaute the Fisher information, other methods can be used
                        loss.backward()
                    for name, param in net.named_parameters():
                        if batch_idx==0:
                            f_local[name] = [(len(target) / len_of_data)*copy.deepcopy(kf_ele)  for kf_ele in param.kfra]
                        else:
                            for kf_idx, kf_ele in enumerate(param.kfra):
                                f_local[name][kf_idx] += (len(target) / len_of_data) * copy.deepcopy(kf_ele)

        if epoch<epochs:
            epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        else:
            for name, param in net.named_parameters():
                # adding I into the Fisher information and minimizing the approximation error from this add
                pi = 1.0
                coor = 1 - coordinator
                if len(f_local[name]) == 2:
                    coor = sqrt(1 - coordinator)
                    # s1 = torch.norm(f_local[name][0], p=2)
                    # s2 = torch.norm(f_local[name][1], p=2)
                    # pi = torch.sqrt(s1/s2)

                for f_idx in range(len(f_local[name])):
                    # f_local[name][f_idx] = torch.clamp(f_local[name][f_idx], -1, 1)
                    if f_idx ==1:
                        pi = 1/pi

                    # f_local[name][f_idx][f_local[name][f_idx]<1e-10] = 0
                    f_local[name][f_idx] = f_local[name][f_idx] + pi*coor*torch.eye(f_local[name][f_idx].shape[0]).to(device)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc

def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map

def newton_solve(fed_avg_freqs, Ak, Bk, Z, device):
    HatM = torch.zeros_like(Z).to(device)
    J = torch.ones_like(Z).to(device)
    GradientF = sum([fed_avg_freqs[idx]*Ak[idx] @ J @ Bk[idx] for idx in range(len(Ak))])
    for i in range(10000):
        fHatM = sum([fed_avg_freqs[idx]*Ak[idx] @ HatM @ Bk[idx] for idx in range(len(Ak))]) - Z
        error = fHatM.abs().mean()
        # if error<1e-7:
        #     break
        HatM = HatM - fHatM/GradientF
    # print(i, " - ", error)
    return HatM

if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                        args.datadir,
                                                                                        args.batch_size,
                                                                                        32)

    print("len train_dl_global:", len(train_ds_global))


    data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)


    if args.alg == 'blockofflinenewton':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets_extend(args.net_config, args.dropout_p, args.n_parties, args)

        global_models, global_model_meta_data, global_layer_type = init_nets_extend(args.net_config, 0, 1, args)
        global_model = global_models[0]
        global_para = global_model.state_dict()

        param_shapes = {}
        fim_shapes = {}

        global_fim = OrderedDict()
        for key, params in global_model.named_parameters():
            global_fim[key] = []

        f_nets = [copy.deepcopy(global_fim) for i in range(args.n_parties)]

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()

            for key in global_model.state_dict():
                param_shapes[key] = global_para[key].shape

            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
                        # nets[idx].__init_net_weights__()
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_blockoffline(nets, selected, f_nets,args, net_dataidx_map, test_dl = test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            avg_param = OrderedDict()
            newtons_param = OrderedDict()
            for idx in range(len(selected)):
                net_para = nets[selected[idx]].state_dict()

                for key in net_para:
                    if idx == 0 and round == 0:
                        if len(param_shapes[key])>2:
                            fim_shapes[key] = [f_nets[selected[idx]][key][0].shape[0], f_nets[selected[idx]][key][1].shape[0]]
                        else:
                            fim_shapes[key] = f_nets[selected[idx]][key][0].shape

                    new_param = copy.deepcopy(net_para[key])

                    # for weights
                    if len(f_nets[selected[idx]][key])==2:

                        if len(param_shapes[key])>2:
                            new_param = (f_nets[selected[idx]][key][0] @ new_param.reshape(fim_shapes[key]) @ f_nets[selected[idx]][key][1]).reshape(param_shapes[key])
                        else:
                            new_param = f_nets[selected[idx]][key][0] @ new_param @ f_nets[selected[idx]][key][1]

                    # for bias
                    elif len(f_nets[selected[idx]][key])==1:
                        new_param = f_nets[selected[idx]][key][0] @ new_param

                    if idx == 0:
                        avg_param[key] = fed_avg_freqs[idx]*net_para[key]
                        global_fim[key] = [fed_avg_freqs[idx]*tem_lf for tem_lf in f_nets[selected[idx]][key]]
                        global_para[key] = new_param * fed_avg_freqs[idx]

                    else:
                        avg_param[key] += fed_avg_freqs[idx]*net_para[key]
                        for tem_idx in range(len(global_fim[key])):
                            global_fim[key][tem_idx] += f_nets[selected[idx]][key][tem_idx] * fed_avg_freqs[idx]
                        global_para[key] += new_param * fed_avg_freqs[idx]

            for key in global_model.state_dict():

                # computing the global expectations
                # for weights
                if len(global_fim[key])==2:
                    # for convolutional weights
                    if len(param_shapes[key])>2:
                        newtons_param[key] = newton_solve(fed_avg_freqs,
                            Ak=[f_nets[selected[idx]][key][0] for idx in range(len(selected))],
                            Bk=[f_nets[selected[idx]][key][1] for idx in range(len(selected))],
                            Z=global_para[key].reshape(fim_shapes[key]),
                            device=device).reshape(param_shapes[key])
                        global_para[key] = (torch.inverse(global_fim[key][0]) @ global_para[key].reshape(fim_shapes[key]) @ torch.inverse(global_fim[key][1])).reshape(param_shapes[key])

                    # for fully connected weights
                    else:
                        newtons_param[key] = newton_solve(fed_avg_freqs,
                                Ak=[f_nets[selected[idx]][key][0] for idx in range(len(selected))],
                                Bk=[f_nets[selected[idx]][key][1] for idx in range(len(selected))],
                                Z=global_para[key],
                                device=device)
                        global_para[key] = torch.inverse(global_fim[key][0]) @ global_para[key] @ torch.inverse(global_fim[key][1])

                # for bias
                elif len(global_fim[key])==1:
                    newtons_param[key] = torch.inverse(global_fim[key][0]) @ global_para[key]
                    global_para[key] = copy.deepcopy(newtons_param[key])

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.load_state_dict(avg_param)
            train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True)
            logger.info('>> fedavg Global Model Train accuracy: %f' % train_acc)
            logger.info('>> fedavg Global Model Test accuracy: %f' % test_acc)
            print('>> fedavg Global Model Train accuracy: %f' % train_acc)
            print('>> fedavg Global Model Test accuracy: %f' % test_acc)

            global_model.load_state_dict(newtons_param)
            train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True)
            logger.info('>> newtons Global Model Train accuracy: %f' % train_acc)
            logger.info('>> newtons Global Model Test accuracy: %f' % test_acc)
            print('>> newtons Global Model Train accuracy: %f' % train_acc)
            print('>> newtons Global Model Test accuracy: %f' % test_acc)

            global_model.load_state_dict(global_para)
            train_acc = compute_accuracy(global_model, train_dl_global)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True)
            logger.info('>> biased Global Model Train accuracy: %f' % train_acc)
            logger.info('>> biased Global Model Test accuracy: %f' % test_acc)
            print('>> biased Global Model Train accuracy: %f' % train_acc)
            print('>> biased Global Model Test accuracy: %f' % test_acc)

            # using newtons' result for the next round
            global_model.load_state_dict(newtons_param)
