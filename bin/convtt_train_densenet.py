#!/usr/bin/env python
import argparse
import logging
import torch
from bidcap.utils.loader import ImagesetLoader, torch_vision_load_cifar10
from convtt.models import densenet
from convtt.train.trainer import *
from cudam import set_visible_gpu

# nohup python convtt_train_densenet.py -g 0 -n 100_12 -d cifar10 >& log/nohup_evaluate_densenet_100_12.log &
# nohup python convtt_train_densenet.py -g 2 -n 100_24 -d cifar10 >& log/nohup_evaluate_densenet_100_24.log &

ARR_AVAILABLE_DENSENETS = [
    '40',
    '100_12',
    '100_24',
    '121',
    '169',
    '201',
    '161'
]

ARR_AVAILABLE_DATASETS = list(ImagesetLoader.dataset_classes().keys())

dropout_rate = 0
weight_decay = 1e-4
momentum = 0.9


def main(args):
    _filter_args(args)
    # configure logging
    log_file_path = 'log/train_densenet_{}.log'.format(args.net_name)
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
    logging.info('===start train densenet. net_name:%s, dataset:%s===', args.net_name, args.dataset)

    # set visible gpu
    if args.gpu_id is not None:set_visible_gpu(args.gpu_id)

    # load dataset
    if args.dataset == 'cifar10':
        dataset = torch_vision_load_cifar10(1)
        image_shape = (3, 32, 32)
    else:
        dataset = ImagesetLoader.load(args.dataset)
        image_shape = dataset.image_shape

    # build model
    model = getattr(densenet, "densenet" + args.net_name, "densenet40")(num_classes=10, image_shape=image_shape,
                                                                        drop_rate=dropout_rate)
    logging.debug('---init weights---')
    model.apply(densenet.init_weights)

    # initialise trainer
    optimiser = build_optimiser(model=model, name='ScheduledSGD', milestones=[150, 225], lr=0.1, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    if args.dataset == 'cifar10':
        driver = build_driver(model=model, training_epoch=300, batch_size=64, optimiser=optimiser,
                              early_stop_max_epochs=300)
        train_loader, test_loader = dataset
        driver._training_loader = train_loader
        driver._validation_loader = test_loader
    else:
        driver = build_driver(model=model, training_epoch=30, batch_size=128, training_data=dataset.train['images'],
                          training_label=dataset.train['labels'],
                          validation_data=None, validation_label=None, test_data=dataset.test['images'],
                          test_label=dataset.test['labels'], optimiser=optimiser)
    if torch.cuda.is_available():
        logging.debug(
            '---the current device cuda:{} is used to train the network---'.format(torch.cuda.current_device()))
    trainer = build_trainer(optimiser=optimiser, driver=driver)

    trainer.driver.train_model()


def _filter_args(args):
    """
    filter the arguments
    :param args: arguments
    """
    args.net_name = str(args.net_name) if args.net_name is not None else None
    args.dataset = str(args.dataset) if args.dataset is not None else None
    args.gpu_id = int(args.gpu_id) if args.gpu_id is not None else None
    if args.net_name not in ARR_AVAILABLE_DENSENETS:
        raise Exception('net: {} is not available'.format(args.net_name))
    if args.dataset not in ARR_AVAILABLE_DATASETS:
        raise Exception('dataset: {} is not available'.format(args.dataset))


# main entrance
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--net_name', help='Densenet extension name, e.g. {}'.format(ARR_AVAILABLE_DENSENETS))
    parser.add_argument('-d', '--dataset', help='Densenet extension name, e.g. {}'.format(ARR_AVAILABLE_DATASETS))
    parser.add_argument('-g', '--gpu_id', help='GPU ID, default: None')

    args = parser.parse_args()
    main(args)
