import os
import shutil
import argparse
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from utils import get_logger, get_new_log_dir, get_checkpoint_path, inf_iterator, load_config, seed_all,\
                  PaddingCollate, recursive_to, get_optimizer, get_scheduler, log_loss, current_milli_time
from data.data import QinghuaDataset
from model import BasicModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--num-gen', type=int, default=50)
    args = parser.parse_args()

    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    log_dir = os.path.dirname(os.path.dirname(args.resume))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    logger = get_logger('evaluate', None)

    logger.info(args)
    logger.info(config)
    logger.info('Loading datasets...')
    train_dataset = QinghuaDataset("data/data.txt")
    logger.info('Dataset length: {} Dictionary length: {}'.format(len(train_dataset), len(train_dataset.dictionary)))
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True,
                              collate_fn=PaddingCollate(train_dataset.dictionary))
    train_iterator = inf_iterator(train_loader)

    model = BasicModel(config.model, train_dataset.dictionary)
    optimizer = get_optimizer(config.train.optimizer, model)

    logger.info('Resuming from checkpoint: %s' % args.resume)
    ckpt = torch.load(args.resume, map_location=args.device)
    it_first = ckpt['iteration']
    model.load_state_dict(ckpt['model'])
    logger.info('Resuming optimizer states...')
    optimizer.load_state_dict(ckpt['optimizer'])

    ret = model.generate(args.num_gen)
    logger.info(ret)
