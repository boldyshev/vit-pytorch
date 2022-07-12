import os

import json

from tqdm import tqdm

import torch
from torch.utils import tensorboard


class Trainer:

    def __init__(self, model, optimizer, criterion, scheduler, model_name):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.model_name = model_name

        self.model_dir = f'models/{self.model_name}'
        self.writer = tensorboard.SummaryWriter(self.model_dir)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.log = dict(training_accuracy=list(),
                        training_loss=list(),
                        validation_accuracy=list(),
                        learning_rate=list())

    def write_log(self, train_acc, epoch_loss, val_acc, lr,  epoch):
        args = (train_acc, epoch_loss, val_acc, lr)
        for key, arg in zip(self.log.keys(), args):
            self.log[key].append(arg)
            self.writer.add_scalar(key, arg, global_step=epoch + 1)

        path = f'{self.model_dir}/log.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.log, f, ensure_ascii=False, indent=4)

    def save_checkpoint(self, epoch):
        """Saves model on epoch 1, 11, 111 to chkpt_ep001.pt, chkpt_ep011.pt, chkpt_ep111.pt respectively"""

        checkpoint_name = 'chkpt_ep' + '0' * (3 - len(str(epoch))) + str(epoch) + '.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, f'{self.model_dir}/checkpoints/{checkpoint_name}')

    def load_checkpoint(self, epoch=None):
        if not epoch:
            checkpoint_name = max(os.listdir(f'{self.model_dir}/checkpoints/'))
        else:
            checkpoint_name = 'chkpt_ep' + '0' * (3 - len(str(epoch))) + str(epoch) + '.pt'
        checkpoint = torch.load(f'{self.model_dir}/checkpoints/' + checkpoint_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint['epoch']

    def save_model(self, log=True):
        path = f'{self.model_dir}/{self.model_name}.pth'
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

        if log:
            path = f'{self.model_dir}/log.json'
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.log, f, ensure_ascii=False, indent=4)

    def train_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        preds = outputs.argmax(dim=-1)
        true_preds = (preds == labels).sum().item()
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item(), true_preds

    def test_step(self, batch):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        preds = self.model(inputs).argmax(dim=-1)
        true_preds = (preds == labels).sum().item()
        return true_preds

    def train(self, train_loader, val_loader, stop_epoch=150, resume=False, log=True, checkpoints=True):
        print(f'{"Resume" if resume else "Start"} training')
        start_epoch = self.load_checkpoint() if resume else 0
        batches_in_dataset = len(train_loader.dataset) // train_loader.batch_size + 1
        num_samples = len(train_loader.dataset)

        for epoch in range(start_epoch, stop_epoch):
            self.model.train()

            # get loss and accuracy
            epoch_loss, train_acc = 0, 0
            for batch in tqdm(train_loader, total=batches_in_dataset):
                loss, true_preds = self.train_step(batch)
                train_acc += true_preds
                epoch_loss += loss
            self.lr_scheduler.step()

            # average over epoch
            train_acc /= num_samples
            epoch_loss /= batches_in_dataset

            # validate model
            self.model.eval()
            val_acc = self.test(val_loader)
            print(f'epoch {epoch}, '
                  f'average loss {epoch_loss:.3f}, '
                  f'training accuracy {train_acc:.3f}, '
                  f'validation accuracy {val_acc:.3f}')

            if log:
                lr = self.lr_scheduler.get_last_lr()[-1]
                self.write_log(train_acc, epoch_loss, val_acc, lr, epoch)
            if checkpoints:
                self.save_checkpoint(epoch)

        print('Finished training')

    def test(self, dataloader):
        true_preds, count = 0., 0
        with torch.no_grad():
            for batch in dataloader:
                true_preds += self.test_step(batch)
        num_samples = len(dataloader.dataset)
        test_acc = true_preds / num_samples

        return test_acc
