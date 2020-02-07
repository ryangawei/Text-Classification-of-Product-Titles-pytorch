# coding=utf-8
import argparse
import logging
import config
import os
import numpy as np
import torch
from title_dataset import TitleDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from nlputils.tokenizer import BasicTokenizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from models import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

assert torch.cuda.is_available()
device = torch.device("cuda:0")

if not os.path.exists(config.LOG_DIR):
    os.mkdir(config.LOG_DIR)
if not os.path.exists(config.CHECKPOINTS_DIR):
    os.mkdir(config.CHECKPOINTS_DIR)


def train(**kwargs):
    if kwargs['mode'] == 'char':
        dataset = np.load(config.TRAIN_CHAR_PATH, allow_pickle=True)
        maxlen = config.MAX_CHAR_TEXT_LENGTH
    else:
        dataset = np.load(config.TRAIN_WORD_PATH, allow_pickle=True)
        maxlen = config.MAX_WORD_TEXT_LENGTH

    x_set = dataset['title_ids']
    y_set = dataset['label_ids']

    x_train, x_valid, y_train, y_valid = train_test_split(x_set, y_set, test_size=0.2,
                                                          shuffle=True, stratify=y_set)

    train_set = TitleDataset(x_train, y_train, maxlen)
    valid_set = TitleDataset(x_valid, y_valid, maxlen)

    logger.info('Train set size: {}, valid set size {}'.format(
        len(train_set), len(valid_set)))

    train_loader = DataLoader(train_set,
                              batch_size=kwargs['batch_size'],
                              shuffle=True)

    valid_loader = DataLoader(valid_set,
                              batch_size=kwargs['batch_size'],
                              shuffle=True)

    if kwargs['model'] == 'textcnn':
        model = TextCNN(kwargs['mode']).cuda(device)
    elif kwargs['model'] == 'textbilstm':
        model = TextBiLSTM(kwargs['mode']).cuda(device)
    elif kwargs['model'] == 'textbigru':
        model = TextBiGRU(kwargs['mode']).cuda(device)

    # List all modules inside the model.
    logger.info('Model modules:')
    for i, m in enumerate(model.named_children()):
        logger.info('{} -> {}'.format(i, m))

    # Get the number of total parameters.
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    logger.info("Total params: {:,}".format(total_params))
    logger.info("Trainable params: {:,}".format(trainable_params))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           patience=8,
                                                           verbose=True
                                                           )

    for epoch in range(kwargs['epoch']):
        # =======================Training===========================
        # Set model to train mode.
        model.train()
        steps = int(np.ceil(len(train_set) // kwargs['batch_size']))
        pbar = tqdm(desc='Epoch {}, loss {}'.format(epoch, 'NAN'),
                    total=steps)
        for i, sample in enumerate(train_loader):
            x, y = sample[0].cuda(device).long(), sample[1].cuda(device).long()
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()
            pbar.set_description(
                'Epoch {}, training loss {:.5f}'.format(epoch, loss.item()))
            pbar.update()
        pbar.close()
        # =========================================================
        # =======================Validation========================
        # Set model to evaluation mode.
        model.eval()
        # Validation step
        valid_loss = 0.0
        valid_acc = 0.0
        for i, sample in enumerate(valid_loader):
            y_true_local = sample[1].numpy()
            x, y_true = sample[0].cuda(
                device).long(), sample[1].cuda(device).long()

            outputs = model(x)
            loss = criterion(outputs, y_true)
            y_pred = outputs.argmax(dim=1).cpu().numpy()
            valid_loss += loss.item()
            valid_acc += accuracy_score(y_true_local, y_pred)
        steps = i + 1
        valid_loss /= steps
        valid_acc /= steps
        
        # Apply ReduceLROnPlateau to the lr.
        scheduler.step(valid_loss)

        logger.info('Epoch {}, valid loss {:.5f}, valid acc {:.4f}'.format(
            epoch, valid_loss, valid_acc))
        # ==========================================================
        # Save the model at the end of every epoch.
        torch.save({
            'model_name': kwargs['model'],
            'epoch': epoch,
            'loss': loss,
            'valid_acc': valid_acc,
            'model_state_dict':  model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }, f=os.path.join(config.CHECKPOINTS_DIR, f'{kwargs["model"]}.pt'))


if __name__ == '__main__':
    model_list = ['textcnn', 'textbilstm', 'textbigru']

    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--lr', default=1e-3,
                        type=float, help='Learning rate. Default 1e-3.')
    parser.add_argument('--batch_size', default=128,
                        type=int, help='Batch size. Default 128.')
    parser.add_argument('--model', default='textbigru', choices=model_list, type=str,
                        help='The classification model. Default textcnn.')
    parser.add_argument('--mode', default='char', type=str,
                        help='Type of the embedding input.')
    parser.add_argument('-e', '--epoch', default=100,
                        type=int, help='Number of training epoch.')

    kwargs = vars(parser.parse_args())

    assert kwargs['model'] in model_list

    formater = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Print logs to the terminal.
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formater)
    # Save logs to file.
    log_path = os.path.join(config.LOG_DIR, f'{kwargs["model"]}_{kwargs["mode"]}.log')
    file_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(formater)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    logger.info('Using device {}'.format(
        torch.cuda.get_device_properties(device)))
    logger.info(kwargs)

    train(**kwargs)
