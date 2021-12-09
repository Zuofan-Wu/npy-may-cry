import os
import shutil
import argparse
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from utils import get_logger, get_new_log_dir, get_checkpoint_path, inf_iterator, load_config, seed_all,\
                  PaddingCollate, recursive_to, get_optimizer, get_scheduler, log_loss, current_milli_time
from data.data import QinghuaDataset
from model import BasicModel, AttentionModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    if args.debug:
        log_dir = None
    else:
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            log_dir = get_new_log_dir(args.logdir, prefix='%s' % config_name)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    logger = get_logger('train', log_dir)

    logger.info(args)
    logger.info(config)
    logger.info('Loading datasets...')
    train_dataset = QinghuaDataset("data/data.txt")
    logger.info('Dataset length: {} Dictionary length: {}'.format(len(train_dataset), len(train_dataset.dictionary)))
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True,
                              collate_fn=PaddingCollate(train_dataset.dictionary))
    train_iterator = inf_iterator(train_loader)

    if config.model.type in ["LSTM", "RNN", "GRU"]:
        model = BasicModel(config.model, train_dataset.dictionary)
    elif config.model.type in ["ATTENTION"]:
        model = AttentionModel(config.model, train_dataset.dictionary)

    optimizer = get_optimizer(config.train.optimizer, model)
    it_first = 0

    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])

    for i in range(it_first, config.train.max_iters):
        data = recursive_to(next(train_iterator), args.device)
        x, mask = data["data"], data["mask"]
        optimizer.zero_grad()

        time_start = current_milli_time()
        loss = model.get_loss(x, mask=mask)
        time_forward_end = current_milli_time()
        loss.backward()
        optimizer.step()
        time_backward_end = current_milli_time()

        log_loss(loss, i, 'train', logger, others={
            'time_forward': (time_forward_end - time_start) / 1000,
            'time_backward': (time_backward_end - time_forward_end) / 1000,
        })
        if (i + 1) % config.train.val_freq == 0:
            with torch.no_grad():
                model.eval()
                ret = model.generate(config.train.val_num)
            model.train()
            logger.info(ret)
            if not args.debug:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % i)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iteration': i,
                }, ckpt_path)