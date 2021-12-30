import numpy as np
import torch
from base import BaseTrainer
from utils.util import MetricTracker


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler

        self.allpoints = data_loader.dataset.allpoints 
        self.train_metrics = MetricTracker('Total_loss', 'loss_pnt_on', 'loss_pnt_in','loss_pnt_out', 'loss_normvec', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('Total_loss', 'loss_pnt_on', 'loss_pnt_in','loss_pnt_out', 'loss_normvec', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (img_H, target) in enumerate(self.data_loader):
            img_H = img_H.to(self.device)
            for key in target:
                if target[key].size==0: print(target['directory']); raise
                target[key] = target[key].to(self.device)

            self.optimizer.zero_grad()
            polycoeff, logits, _, _ = self.model(img_H) # data: images, output: (params, R, logits)
            loss, loss_valdict, PI_value_inout = self.criterion(polycoeff, target, self.config['loss_weights'])
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.train_metrics.update('Total_loss', loss.item())
            for los in loss_valdict:
                self.train_metrics.update(los, loss_valdict[los])
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(PI_value_inout, target, False))

            if batch_idx % int(np.sqrt(self.data_loader.batch_size)) == 0:
                self.logger.debug('Train Epoch: {} {} TotalLoss: {:.6f} = {:.6g} + {:.6g} + {:.6g}, normvec: {:.5g}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item(),
                    loss_valdict['loss_pnt_on'], 
                    loss_valdict['loss_pnt_in'], 
                    loss_valdict['loss_pnt_out'], 
                    loss_valdict['loss_normvec']
                ))

            del img_H; del polycoeff; del logits; del loss

        log = self.train_metrics.result()

        if self.valid_data_loader is not None:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.writer.close()
        
        return log


    def _valid_epoch(self, epoch):
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (img_H, target) in enumerate(self.valid_data_loader):
                img_H = img_H.to(self.device)
                for key in target:
                    if target[key].size==0: print(target['directory']); raise
                    target[key] = target[key].to(self.device)

                polycoeff, _, _, _ = self.model(img_H)
                loss, loss_valdict, PI_value_inout = self.criterion(polycoeff, target, self.config['loss_weights'])

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                
                self.valid_metrics.update('Total_loss', loss.item())
                for los in loss_valdict:
                    self.valid_metrics.update(los, loss_valdict[los])
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(PI_value_inout, target, False))

                del img_H; del loss

        self.writer.close()
        return self.valid_metrics.result()


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = len(self.data_loader)
        return base.format(current, total, 100.0 * current / total)

