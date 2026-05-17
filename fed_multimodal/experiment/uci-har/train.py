import json
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path

from fed_multimodal.constants import constants
from fed_multimodal.trainers.server_trainer import Server
from fed_multimodal.model.mm_models import HARClassifier
from fed_multimodal.model.unimodal_models import ConvRNNClassifier
from fed_multimodal.dataloader.dataload_manager import DataloadManager
from fed_multimodal.dataloader.local_eval_split import (
    save_local_eval_split_json,
    split_multimodal_client_records,
    split_unimodal_client_records,
)

from fed_multimodal.trainers.fed_rs_trainer import ClientFedRS
from fed_multimodal.trainers.fed_avg_trainer import ClientFedAvg
from fed_multimodal.trainers.scaffold_trainer import ClientScaffold
from fed_multimodal.trainers.client_metrics import (
    append_client_metrics,
    build_client_metrics_row,
)
from fed_multimodal.trainers.per_client_eval_metrics import (
    build_per_client_eval_row,
    write_per_client_eval_rows,
)

# Define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(device_arg):
    device_arg = str(device_arg).strip().lower()

    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    if device_arg == "cpu":
        return torch.device("cpu")

    if device_arg == "cuda":
        device_arg = "cuda:0"

    if device_arg.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA device '{device_arg}' was requested, but CUDA is not available.")

        try:
            device_idx = int(device_arg.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"Invalid CUDA device '{device_arg}'. Expected format: cuda:N, e.g. cuda:0.")

        device_count = torch.cuda.device_count()
        if device_idx < 0 or device_idx >= device_count:
            raise ValueError(
                f"CUDA device '{device_arg}' is unavailable. "
                f"Detected {device_count} CUDA device(s); valid ids are 0 to {device_count - 1}."
            )
        return torch.device(device_arg)

    raise ValueError("Invalid --device value. Use one of: auto, cpu, cuda, cuda:N.")


def parse_fold(fold_arg):
    try:
        fold_idx = int(fold_arg)
    except ValueError:
        raise argparse.ArgumentTypeError("fold must be an integer from 1 to 5.")

    if fold_idx < 1 or fold_idx > 5:
        raise argparse.ArgumentTypeError("fold must be an integer from 1 to 5.")
    return fold_idx


def parse_local_eval_ratio(ratio_arg):
    ratio = float(ratio_arg)
    if ratio <= 0 or ratio >= 1:
        raise argparse.ArgumentTypeError("local_eval_ratio must satisfy 0 < ratio < 1.")
    return ratio


def parse_positive_int(value_arg):
    value = int(value_arg)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer.")
    return value


def subset_sim_dict(client_sim_dict, indices):
    if client_sim_dict is None:
        return None
    return [copy.deepcopy(client_sim_dict[idx]) for idx in indices]


def normalize_modality(modality):
    if modality == "acc_gyro":
        return "multimodal"
    return modality


def validate_modality_args(args):
    if args.modality == "multimodal":
        logging.warning("--modality multimodal is deprecated; use --modality acc_gyro.")
        args.modality = "acc_gyro"

    if args.modality in ["acc", "gyro"]:
        if args.att and args.att_name != "base":
            raise ValueError("Only --att_name base is supported for acc/gyro single-modality training.")
        if args.missing_modality:
            raise ValueError(
                "Missing-modality simulation is not supported for pure acc/gyro single-modality training."
            )


def parse_args():
    # read path config files
    path_conf = dict()
    with open(str(Path(os.path.realpath(__file__)).parents[2].joinpath('system.cfg'))) as f:
        for line in f:
            key, val = line.strip().split('=')
            path_conf[key] = val.replace("\"", "")
            
    # If default setting
    if path_conf["data_dir"] == ".":
        path_conf["data_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('data'))
    if path_conf["output_dir"] == ".":
        path_conf["output_dir"] = str(Path(os.path.realpath(__file__)).parents[2].joinpath('output'))

    parser = argparse.ArgumentParser(description='FedMultimoda experiments')
    parser.add_argument(
        '--data_dir', 
        default=path_conf["output_dir"],
        type=str, 
        help='output feature directory'
    )

    parser.add_argument(
        '--device',
        default='auto',
        type=str,
        help='device to use: auto, cpu, cuda, or cuda:N',
    )

    parser.add_argument(
        '--fold',
        default=None,
        type=parse_fold,
        help='optional fold index to run; defaults to all folds 1-5',
    )

    parser.add_argument(
        '--save_client_metrics',
        action='store_true',
        help='save per-client train-time metrics to CSV',
    )

    parser.add_argument(
        '--client_metrics_dir',
        default=None,
        type=str,
        help='optional directory for client_metrics.csv',
    )

    parser.add_argument(
        '--save_per_client_eval',
        action='store_true',
        help='save per-client final evaluation metrics to CSV',
    )

    parser.add_argument(
        '--per_client_eval_stage',
        default='final',
        choices=['final'],
        help='when to run per-client evaluation; currently only final is supported',
    )

    parser.add_argument(
        '--per_client_eval_dir',
        default=None,
        type=str,
        help='optional directory for per_client_eval_metrics.csv',
    )

    parser.add_argument(
        '--per_client_eval_data',
        default='local_train',
        choices=['local_train', 'local_eval'],
        help='which client data to evaluate',
    )

    parser.add_argument(
        '--enable_local_eval_split',
        action='store_true',
        help='split each client into local_train/local_eval records',
    )

    parser.add_argument(
        '--local_eval_ratio',
        default=0.2,
        type=parse_local_eval_ratio,
        help='ratio of each client held out for local evaluation',
    )

    parser.add_argument(
        '--local_eval_seed',
        default=2026,
        type=int,
        help='base random seed for local evaluation split',
    )

    parser.add_argument(
        '--local_eval_min_samples',
        default=1,
        type=parse_positive_int,
        help='minimum local eval samples for each splittable client',
    )

    parser.add_argument(
        '--save_local_eval_split',
        action='store_true',
        help='save local eval split metadata; always saved when local eval split is enabled',
    )
    
    parser.add_argument(
        '--acc_feat', 
        default='acc',
        type=str,
        help="acc feature name",
    )
    
    parser.add_argument(
        '--gyro_feat', 
        default='gyro',
        type=str,
        help="gyro feature name",
    )
    
    parser.add_argument(
        '--learning_rate', 
        default=0.05,
        type=float,
        help="learning rate",
    )
    
    parser.add_argument(
        '--global_learning_rate', 
        default=0.05,
        type=float,
        help="learning rate",
    )
    
    parser.add_argument(
        '--sample_rate', 
        default=0.1,
        type=float,
        help="client sample rate",
    )
    
    parser.add_argument(
        '--num_epochs', 
        default=300,
        type=int,
        help="total training rounds",
    )

    parser.add_argument(
        '--test_frequency', 
        default=1,
        type=int,
        help="perform test frequency",
    )
    
    parser.add_argument(
        '--local_epochs', 
        default=1,
        type=int,
        help="local epochs",
    )
    
    parser.add_argument(
        '--hid_size',
        type=int, 
        default=64,
        help='RNN hidden size dim'
    )
    
    parser.add_argument(
        '--optimizer', 
        default='sgd',
        type=str,
        help="optimizer",
    )

    parser.add_argument(
        '--mu',
        type=float, 
        default=0.001,
        help='Fed prox term'
    )
    
    parser.add_argument(
        '--fed_alg', 
        default='fed_avg',
        type=str,
        help="federated learning aggregation algorithm",
    )
    
    parser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help="training batch size",
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha in direchlet distribution",
    )
    
    parser.add_argument(
        '--att', 
        type=bool, 
        default=False,
        help='self attention applied or not'
    )
    
    parser.add_argument(
        "--en_att",
        dest='att',
        action='store_true',
        help="enable self-attention"
    )
    
    parser.add_argument(
        '--att_name',
        type=str, 
        default='multihead',
        help='attention name'
    )
    
    parser.add_argument(
        "--missing_modality",
        type=bool, 
        default=False,
        help="missing modality simulation",
    )
    
    parser.add_argument(
        "--en_missing_modality",
        dest='missing_modality',
        action='store_true',
        help="enable missing modality simulation",
    )
    
    parser.add_argument(
        "--missing_modailty_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        "--missing_label",
        type=bool, 
        default=False,
        help="missing label simulation",
    )
    
    parser.add_argument(
        "--en_missing_label",
        dest='missing_label',
        action='store_true',
        help="enable missing label simulation",
    )
    
    parser.add_argument(
        "--missing_label_rate",
        type=float, 
        default=0.5,
        help='missing rate for modality; 0.9 means 90%% missing'
    )
    
    parser.add_argument(
        '--label_nosiy', 
        type=bool, 
        default=False,
        help='clean label or nosiy label')
    
    parser.add_argument(
        "--en_label_nosiy",
        dest='label_nosiy',
        action='store_true',
        help="enable label noise simulation",
    )

    parser.add_argument(
        '--label_nosiy_level', 
        type=float, 
        default=0.1,
        help='nosiy level for labels; 0.9 means 90% wrong'
    )
    
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="uci-har",
        help='data set name'
    )

    parser.add_argument(
        '--modality', 
        type=str, 
        default='acc_gyro',
        choices=['acc_gyro', 'acc', 'gyro', 'multimodal'],
        help='modality type'
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    # argument parser
    args = parse_args()
    validate_modality_args(args)
    args.modality = normalize_modality(args.modality)
    if args.per_client_eval_data == "local_eval" and not args.enable_local_eval_split:
        raise ValueError("--per_client_eval_data local_eval requires --enable_local_eval_split.")

    # data manager
    dm = DataloadManager(args)
    dm.get_simulation_setting(alpha=args.alpha)
    
    # find device
    device = resolve_device(args.device)
    logging.info(f'Using device: {device}')
    if device.type == 'cuda': logging.info(f'GPU available, use GPU: {torch.cuda.get_device_name(device.index)}')
    save_result_dict = dict()
    fold_indices = list(range(1, 6)) if args.fold is None else [args.fold]
    if args.fold is None:
        logging.info('Running all folds: fold1-fold5')
    else:
        logging.info(f'Running fold{args.fold}')

    if args.fed_alg in ['fed_avg', 'fed_prox', 'fed_opt']:
        Client = ClientFedAvg
    elif args.fed_alg in ['scaffold']:
        Client = ClientScaffold
    elif args.fed_alg in ['fed_rs']:
        Client = ClientFedRS

    # load simulation feature
    dm.load_sim_dict()
    # load client ids
    dm.get_client_ids()
    # load feature records
    acc_record_dict = dict()
    gyro_record_dict = dict()
    unimodal_record_dict = dict()
    logging.info('Reading Data')
    for client_id in tqdm(dm.client_ids):
        if args.modality == "multimodal":
            acc_dict = dm.load_acc_feat(
                client_id=client_id
            )
            gyro_dict = dm.load_gyro_feat(
                client_id=client_id
            )
            dm.get_label_dist(
                gyro_dict, 
                client_id
            )
            acc_record_dict[client_id] = acc_dict
            gyro_record_dict[client_id] = gyro_dict
        elif args.modality == "acc":
            data_dict = dm.load_acc_feat(
                client_id=client_id
            )
            dm.get_label_dist(
                data_dict,
                client_id
            )
            unimodal_record_dict[client_id] = data_dict
        elif args.modality == "gyro":
            data_dict = dm.load_gyro_feat(
                client_id=client_id
            )
            dm.get_label_dist(
                data_dict,
                client_id
            )
            unimodal_record_dict[client_id] = data_dict
    
    # We perform 5 fold experiments with 5 seeds by default.
    for fold_idx in fold_indices:
        # number of clients
        client_ids = [client_id for client_id in dm.client_ids if client_id not in ['dev', 'test']]
        num_of_clients = len(client_ids)
        
        # set seeds
        set_seed(8*fold_idx)
        # loss function
        criterion = nn.NLLLoss().to(device)
        # Define the model
        if args.modality == "multimodal":
            global_model = HARClassifier(
                num_classes=constants.num_class_dict[args.dataset],         # Number of classes
                acc_input_dim=constants.feature_len_dict[args.acc_feat],    # Acc data input dim
                gyro_input_dim=constants.feature_len_dict[args.gyro_feat],  # Gyro data input dim
                en_att=args.att,                                            # Enable self attention or not
                d_hid=args.hid_size,
                att_name=args.att_name
            )
        else:
            global_model = ConvRNNClassifier(
                num_classes=constants.num_class_dict[args.dataset],
                input_dim=constants.feature_len_dict[args.modality],
                d_hid=args.hid_size,
                en_att=args.att,
                att_name=args.att_name
            )
        global_model = global_model.to(device)

        # initialize server
        server = Server(
            args, 
            global_model, 
            device=device, 
            criterion=criterion,
            client_ids=client_ids
        )
        server.initialize_log(fold_idx)
        server.sample_clients(
            num_of_clients, 
            sample_rate=args.sample_rate
        )
        server.get_num_params()

        # save json path
        save_json_path = Path(os.path.realpath(__file__)).parents[2].joinpath(
            'result', 
            args.fed_alg,
            args.dataset,
            server.feature,
            server.att,
            server.model_setting_str
        )
        Path.mkdir(save_json_path, parents=True, exist_ok=True)
        if args.client_metrics_dir is None:
            client_metrics_path = save_json_path.joinpath("client_metrics.csv")
        else:
            client_metrics_path = Path(args.client_metrics_dir).joinpath("client_metrics.csv")
        if args.save_client_metrics:
            logging.info(f'Saving client-level metrics to {client_metrics_path}')
        if args.per_client_eval_dir is None:
            per_client_eval_metrics_path = save_json_path.joinpath("per_client_eval_metrics.csv")
        else:
            per_client_eval_metrics_path = Path(args.per_client_eval_dir).joinpath("per_client_eval_metrics.csv")
        if args.save_per_client_eval:
            logging.info(f'Saving per-client evaluation metrics to {per_client_eval_metrics_path}')
        local_eval_split_path = save_json_path.joinpath("local_eval_split.json")

        server.save_json_file(
            dm.label_dist_dict, 
            save_json_path.joinpath('label.json')
        )

        dataloader_dict = dict()
        local_eval_dataloader_dict = dict()
        local_eval_split_info = dict()
        local_eval_split_seed = args.local_eval_seed + 8 * fold_idx
        for client_id in dm.client_ids:
            shuffle = False if client_id in ['dev', 'test'] else True
            client_sim_dict = None if client_id in ['dev', 'test'] else dm.get_client_sim_dict(client_id=client_id)

            if args.modality == "multimodal":
                acc_records = acc_record_dict[client_id]
                gyro_records = gyro_record_dict[client_id]

                if args.enable_local_eval_split and client_id not in ['dev', 'test']:
                    (
                        acc_train,
                        gyro_train,
                        acc_eval,
                        gyro_eval,
                        split_info,
                    ) = split_multimodal_client_records(
                        acc_records,
                        gyro_records,
                        eval_ratio=args.local_eval_ratio,
                        seed=local_eval_split_seed,
                        min_eval_samples=args.local_eval_min_samples,
                        client_id=client_id,
                    )
                    local_eval_split_info[client_id] = split_info
                    train_sim_dict = subset_sim_dict(client_sim_dict, split_info["train_indices"])
                    eval_sim_dict = subset_sim_dict(client_sim_dict, split_info["eval_indices"])
                    dm.get_label_dist(
                        gyro_train,
                        client_id
                    )
                    dataloader_dict[client_id] = dm.set_dataloader(
                        acc_train,
                        gyro_train,
                        shuffle=shuffle,
                        client_sim_dict=train_sim_dict,
                        default_feat_shape_a=np.array([128, constants.feature_len_dict[args.acc_feat]]),
                        default_feat_shape_b=np.array([128, constants.feature_len_dict[args.gyro_feat]]),
                    )
                    if len(acc_eval) == 0:
                        local_eval_dataloader_dict[client_id] = None
                    else:
                        local_eval_dataloader_dict[client_id] = dm.set_dataloader(
                            acc_eval,
                            gyro_eval,
                            shuffle=False,
                            client_sim_dict=eval_sim_dict,
                            default_feat_shape_a=np.array([128, constants.feature_len_dict[args.acc_feat]]),
                            default_feat_shape_b=np.array([128, constants.feature_len_dict[args.gyro_feat]]),
                        )
                else:
                    dataloader_dict[client_id] = dm.set_dataloader(
                        acc_records,
                        gyro_records,
                        shuffle=shuffle,
                        client_sim_dict=client_sim_dict,
                        default_feat_shape_a=np.array([128, constants.feature_len_dict[args.acc_feat]]),
                        default_feat_shape_b=np.array([128, constants.feature_len_dict[args.gyro_feat]]),
                    )
            else:
                records = unimodal_record_dict[client_id]
                if args.enable_local_eval_split and client_id not in ['dev', 'test']:
                    train_records, eval_records, split_info = split_unimodal_client_records(
                        records,
                        eval_ratio=args.local_eval_ratio,
                        seed=local_eval_split_seed,
                        min_eval_samples=args.local_eval_min_samples,
                        client_id=client_id,
                    )
                    local_eval_split_info[client_id] = split_info
                    dm.get_label_dist(
                        train_records,
                        client_id
                    )
                    dataloader_dict[client_id] = dm.set_unimodal_dataloader(
                        train_records,
                        shuffle=shuffle,
                    )
                    if len(eval_records) == 0:
                        local_eval_dataloader_dict[client_id] = None
                    else:
                        local_eval_dataloader_dict[client_id] = dm.set_unimodal_dataloader(
                            eval_records,
                            shuffle=False,
                        )
                else:
                    dataloader_dict[client_id] = dm.set_unimodal_dataloader(
                        records,
                        shuffle=shuffle,
                    )

        if args.enable_local_eval_split:
            split_json_dict = {
                "dataset": args.dataset,
                "modality": server.feature,
                "alpha": args.alpha,
                "fold": fold_idx,
                "local_eval_ratio": args.local_eval_ratio,
                "local_eval_seed": args.local_eval_seed,
                "effective_local_eval_seed": local_eval_split_seed,
                "local_eval_min_samples": args.local_eval_min_samples,
                "clients": local_eval_split_info,
            }
            save_local_eval_split_json(split_json_dict, local_eval_split_path)
            logging.info(f'Saved local eval split metadata to {local_eval_split_path}')
        
        # set seeds again
        set_seed(8*fold_idx)
        # Training steps
        for epoch in range(int(args.num_epochs)):
            # define list varibles that saves the weights, loss, num_sample, etc.
            server.initialize_epoch_updates(epoch)
            # 1. Local training, return weights in fed_avg, return gradients in fed_sgd
            skip_client_ids = list()
            for idx in server.clients_list[epoch]:
                # Local training
                client_id = client_ids[idx]
                dataloader = dataloader_dict[client_id]
                if dataloader is None:
                    skip_client_ids.append(client_id)
                    continue

                # Initialize client object
                client = Client(
                    args, 
                    device, 
                    criterion, 
                    dataloader, 
                    model=copy.deepcopy(server.global_model),
                    label_dict=dm.label_dist_dict[client_id],
                    num_class=constants.num_class_dict[args.dataset]
                )

                if args.fed_alg == 'scaffold':
                    client.set_control(
                        server_control=copy.deepcopy(server.server_control), 
                        client_control=copy.deepcopy(server.client_controls[client_id])
                    )
                    client.update_weights()

                    # server append updates
                    server.set_client_control(client_id, copy.deepcopy(client.client_control))
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()), 
                        client.result['sample'], 
                        client.result,
                        delta_control=copy.deepcopy(client.delta_control)
                    )
                else:
                    client.update_weights()
                    # server append updates
                    server.save_train_updates(
                        copy.deepcopy(client.get_parameters()), 
                        client.result['sample'], 
                        client.result
                    )
                if args.save_client_metrics:
                    append_client_metrics(
                        client_metrics_path,
                        build_client_metrics_row(
                            args=args,
                            fold_idx=fold_idx,
                            epoch=epoch,
                            client_id=client_id,
                            client_result=client.result,
                            modality_setting=server.feature,
                        )
                    )
                del client
            
            # logging skip client
            logging.info(f'Client Round: {epoch}, Skip client {skip_client_ids}')
            
            # 2. aggregate, load new global weights
            if len(server.num_samples_list) == 0: continue
            server.average_weights()
            logging.info('---------------------------------------------------------')
            server.log_classification_result(
                data_split='train',
                metric='f1'
            )
            if epoch % args.test_frequency == 0:
                with torch.no_grad():
                    # 3. Perform the validation on dev set
                    server.inference(dataloader_dict['dev'])
                    server.result_dict[epoch]['dev'] = server.result
                    server.log_classification_result(
                        data_split='dev',
                        metric='f1'
                    )

                    # 4. Perform the test on holdout set
                    server.inference(dataloader_dict['test'])
                    server.result_dict[epoch]['test'] = server.result
                    server.log_classification_result(
                        data_split='test',
                        metric='f1'
                    )
                
                logging.info('---------------------------------------------------------')
                server.log_epoch_result(metric='f1')
            logging.info('---------------------------------------------------------')

        # Performance save code
        save_result_dict[f'fold{fold_idx}'] = server.summarize_dict_results()
        
        # output to results
        server.save_json_file(
            save_result_dict, 
            save_json_path.joinpath('result.json')
        )

        if args.save_per_client_eval:
            per_client_eval_rows = list()
            final_epoch = int(args.num_epochs) - 1
            with torch.no_grad():
                for client_id in client_ids:
                    if args.per_client_eval_data == "local_eval":
                        dataloader = local_eval_dataloader_dict.get(client_id)
                        eval_data_type = "local_eval_split"
                        split_info = local_eval_split_info.get(client_id)
                        split_file = str(local_eval_split_path)
                    else:
                        dataloader = dataloader_dict.get(client_id)
                        eval_data_type = "local_train_eval"
                        split_info = None
                        split_file = None
                    if dataloader is None:
                        logging.warning(f'Skip per-client eval for {client_id}: dataloader is None')
                        continue

                    server.inference(dataloader)
                    per_client_eval_rows.append(
                        build_per_client_eval_row(
                            args=args,
                            fold_idx=fold_idx,
                            epoch=final_epoch,
                            eval_stage=args.per_client_eval_stage,
                            client_id=client_id,
                            eval_result=copy.deepcopy(server.result),
                            modality_setting=server.feature,
                            eval_data_type=eval_data_type,
                            split_info=split_info,
                            split_file=split_file,
                        )
                    )
            write_per_client_eval_rows(
                per_client_eval_metrics_path,
                per_client_eval_rows,
            )
            logging.info(
                f'Saved {len(per_client_eval_rows)} per-client evaluation rows to '
                f'{per_client_eval_metrics_path}'
            )

    # Calculate the average of the 5-fold experiments
    save_result_dict['average'] = dict()
    for metric in ['f1', 'acc', 'top5_acc']:
        result_list = list()
        for key in save_result_dict:
            if metric not in save_result_dict[key]: continue
            result_list.append(save_result_dict[key][metric])
        save_result_dict['average'][metric] = np.nanmean(result_list)
    
    # dump the dictionary
    server.save_json_file(
        save_result_dict, 
        save_json_path.joinpath('result.json')
    )
