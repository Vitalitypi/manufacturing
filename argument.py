import argparse
import configparser


def get_arguments(dataset,batch_size=1):
    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('--dataset', default=dataset, type=str)
    args.add_argument('--mode', default='test', type=str)
    args.add_argument('--device', default='cuda:1', type=str, help='indices of GPUs')
    args.add_argument('--debug', default='False', type=eval)
    args.add_argument('--model', default='MGSTGNN', type=str)
    args.add_argument('--cuda', default=True, type=bool)

    config_file = './config/{}.conf'.format(dataset)
    config = configparser.ConfigParser()
    config.read(config_file)
    #data
    args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    args.add_argument('--in_steps', default=config['data']['in_steps'], type=int)
    args.add_argument('--out_steps', default=config['data']['out_steps'], type=int)
    args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    args.add_argument('--adj_norm', default=config['data']['adj_norm'], type=eval)
    #model
    args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    args.add_argument('--num_input_dim', default=config['model']['num_input_dim'], type=int)
    args.add_argument('--periods_embedding_dim', default=config['model']['periods_embedding_dim'], type=int)
    args.add_argument('--weekend_embedding_dim', default=config['model']['weekend_embedding_dim'], type=int)
    args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
    args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
    args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
    args.add_argument('--periods', default=config['model']['periods'], type=int)
    args.add_argument('--weekend', default=config['model']['weekend'], type=int)
    args.add_argument('--kernel', default=config['model']['kernel'], type=int)
        
    #train
    args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    args.add_argument('--random', default=config['train']['random'], type=eval)
    args.add_argument('--seed', default=config['train']['seed'], type=int)
    args.add_argument('--batch_size', default=batch_size, type=int)
    args.add_argument('--epochs', default=config['train']['epochs'], type=int)
    args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')

    #test
    args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
    #log
    args.add_argument('--log_dir', default='./', type=str)
    args.add_argument('--log_step', default=config['log']['log_step'], type=int)
    args.add_argument('--plot', default=config['log']['plot'], type=eval)
    args = args.parse_args()
    return args