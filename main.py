import torch
from easydict import EasyDict as edict

import dataset
from utils import AverageMeter, clean_print
from model import Encoder


## OPTIMIZERS
def get_optimizer(cfg, model):
    model_lr = cfg.lr
    encoder_parameters = filter(lambda p: p.requires_grad, model.parameters())  # exclude the embedding layer
    optimizer = torch.optim.Adam(encoder_parameters, lr=model_lr)
    return optimizer


def load_model_and_optimizer(cfg):
    model = Encoder(cfg).to(cfg.device)
    optimizer = get_optimizer(cfg, model)
    return model, optimizer


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    loss_record = AverageMeter()
    acc_record = AverageMeter()
    num_batches = len(train_loader)
    for i_batch, batch in enumerate(train_loader):
        questions, labels = batch
        questions = questions.to(cfg.device)
        labels = labels.to(cfg.device)

        labels_pred = model(questions)
        acc = calc_accuracy(labels_pred, labels)

        loss = criterion(labels_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_samples = len(labels)
        acc_record.update(acc, num_samples)
        loss_record.update(loss, num_samples)
        clean_print('Train Epoch: {0}\t Iteration: {1}/{2}\t Precision: {3:.5f}\t'
                    .format(epoch, i_batch + 1, num_batches, acc_record.avg), end='\r')

    clean_print('Train Epoch: {0}\t Average Train Precision: {1:.5f}\t Average Train Loss: {2:.5f}'
                .format(epoch, acc_record.avg, loss_record.avg))

    return acc_record.avg, loss_record.avg


#
# def calc_precision(labels_pred, labels):
#     labels_pred = labels_pred.argmax(dim=1)
#     accuracy = (labels_pred == labels).sum().item() / len(labels)
#     return accuracy


def calc_accuracy(labels_pred, labels):
    labels = labels.detach().cpu().numpy()
    labels_pred = labels_pred.detach().cpu().numpy()

    labels_pred = (labels_pred > 0.5)
    correct = labels_pred == labels
    acc = correct.sum() / float(len(labels))
    return acc


def val(val_loader, model, epoch):
    model.eval()
    acc_record = AverageMeter()
    num_batches = len(val_loader)

    for i_batch, batch in enumerate(val_loader):
        questions, labels = batch
        questions = questions.to(cfg.device)
        labels = labels.to(cfg.device)

        labels_pred = model(questions)
        acc = calc_accuracy(labels_pred, labels)

        num_samples = len(labels)
        acc_record.update(acc, num_samples)

        clean_print('Val Epoch: {0}\t Iteration: {1}/{2}\t Precision: {3:.5f}\t'
                    .format(epoch, i_batch + 1, num_batches, acc_record.avg), end='\r')

    clean_print('Val Epoch: {0}\t Average Val Precision: {1:.5f}\t Iterations: {2}'
                .format(epoch, acc_record.avg, num_batches))

    return acc_record.avg


def test(test_loader, model):
    labels_predicted = []

    def add_to_labels(labels_pred):
        labels_pred_np = list((labels_pred.detach().cpu().numpy() > 0.5).astype(int))
        labels_predicted.extend(labels_pred_np)

    model.eval()
    for i_batch, questions in enumerate(test_loader):
        questions = questions.to(cfg.device)
        add_to_labels(model(questions))

    return labels_predicted


def dump_test_results(fn, labels_predicted):
    labels_predicted_str = list(map(str, labels_predicted))
    with open(fn, 'w') as f_:
        f_.write('\n'.join(labels_predicted_str))
    print('Results written to {}'.format(fn))


def main(cfg):
    train_loader, val_loader, test_dataloader = dataset.get_dataloaders(cfg)
    model, optimizer = load_model_and_optimizer(cfg)
    criterion = torch.nn.BCELoss()

    num_epochs = cfg.num_epoch

    print('\nTraining started...\n')

    for epoch in range(num_epochs):
        val(val_loader, model, epoch)
        train(train_loader, model, criterion, optimizer, epoch)

    predicted = test(test_dataloader, model)
    dump_test_results(cfg.test_dump_fn, predicted)


if __name__ == '__main__':
    cfg = edict()
    cfg.device = "cuda:0"

    # train params
    cfg.batch_size = 10
    cfg.num_workers = 0
    cfg.num_epoch = 2
    cfg.lr = 0.001
    cfg.val_frequency = 5

    # model params
    cfg.lstm_dropout = 0.4
    cfg.lstm_dim = 150
    cfg.bidirectional = False
    cfg.train_embedding = True
    cfg.T = 15
    cfg.num_vocab = 72704
    cfg.embed_dim = 300
    cfg.num_layers = 1
    cfg.word_count = 15

    # directories and files
    cfg.train_qdb_fn = '../coqa/train_db'
    cfg.val_qdb_fn = '../coqa/val_db'
    cfg.test_qdb_fn = '../test/test_db'

    cfg.embed_matrix_fn = '../word_embedding/embed_matrix.npy'
    cfg.vocab_file_fn = '../word_embedding/vocabulary_72700.txt'

    cfg.test_dump_fn = '../test/test_results'

    main(cfg)
