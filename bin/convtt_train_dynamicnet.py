#!/usr/bin/env python
import argparse
import logging
from bidcap.utils.loader import ImagesetLoader
from convtt.models.dynamicnet import dynamicnet
from convtt.train.trainer import *


ARR_AVAILABLE_DATASETS = list(ImagesetLoader.dataset_classes().keys())


def main(args):
    _filter_args(args)
    # configure logging
    log_file_path = 'log/train_dynamicnet.log'
    logging.basicConfig(filename=log_file_path, level=logging.DEBUG)
    logging.info('===start train dynamicnet on dataset:%s===', args.dataset)

    # load dataset
    dataset = ImagesetLoader.load(args.dataset)
    image_shape = dataset.image_shape
    # build model
    model = dynamicnet(image_shape=image_shape)

    # initialise trainer
    optimiser = build_optimiser(model=model, name='ScheduledSGD', milestones=[10, 20], lr=0.1)
    driver = build_driver(model=model, training_epoch=5, batch_size=128, training_data=dataset.train['images'],
                          training_label=dataset.train['labels'],
                          validation_data=None, validation_label=None, test_data=dataset.test['images'],
                          test_label=dataset.test['labels'], optimiser=optimiser)
    trainer = build_trainer(optimiser=optimiser, driver=driver)
    test_acc = trainer.eval()
    print(test_acc)


def _filter_args(args):
    """
    filter the arguments
    :param args: arguments
    """
    args.dataset = str(args.dataset) if args.dataset is not None else None
    if args.dataset not in ARR_AVAILABLE_DATASETS:
        raise Exception('dataset: {} is not available'.format(args.dataset))


# main entrance
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', help='Densenet extension name, e.g. {}'.format(ARR_AVAILABLE_DATASETS))
    args = parser.parse_args()
    main(args)
