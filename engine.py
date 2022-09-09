import os
import shutil
import torch
from datasets import get_dataset
from models import get_model
from losses import get_loss
from utils.meters import AverageMeter
from utils.transforms import get_train_transform, get_val_transform
from utils.visualizer import ObjectDetectionVisualizer
from utils.collate import box_collate_fn
from utils.parse_detections import parse_detections, parse_batch_detections
from utils.ap_metrics import APMeter
from tqdm import tqdm

class Engine:

    def __init__(self, config, device=None):

        self.config = config
        self.device = device if device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # create output dir
        if config.save:
            os.makedirs(config.save_location, exist_ok=True)

        # train/val dataloaders
        self.train_dataset_it, self.val_dataset_it = self.get_dataloaders(config, self.device)
        
        # model
        self.model = self.get_model(config, self.device)
        
        # loss
        self.loss_fn = self.get_loss(config)
        
        # optimizer/scheduler
        self.optimizer, self.scheduler = self.get_optimizer_and_scheduler(config, self.model)

        # visualizer 
        self.visualizer = ObjectDetectionVisualizer()

    @staticmethod
    def get_dataloaders(config, device):
        train_transform = get_train_transform(config.transform)
        train_dataset = get_dataset(
            config.train_dataset.name, config.train_dataset.params)
        train_dataset.transform = train_transform
        train_dataset_it = torch.utils.data.DataLoader(
            train_dataset, collate_fn=box_collate_fn, batch_size=config.train_dataset.batch_size, shuffle=True, drop_last=True, num_workers=config.train_dataset.num_workers, pin_memory=True if device.type == 'cuda' else False)

        # val dataloader
        val_transform = get_val_transform(config.transform)
        val_dataset = get_dataset(
            config.val_dataset.name, config.val_dataset.params)
        val_dataset.transform = val_transform
        val_dataset_it = torch.utils.data.DataLoader(
            val_dataset, collate_fn=box_collate_fn, batch_size=config.val_dataset.batch_size, shuffle=False, drop_last=False, num_workers=config.val_dataset.num_workers, pin_memory=True if device.type == 'cuda' else False)

        return train_dataset_it, val_dataset_it

    @staticmethod
    def get_model(config, device):
        model = get_model(config.model.name, config.model.params).to(device)

        # load checkpoint
        if config.pretrained_model is not None and os.path.exists(config.pretrained_model):
            print(f'Loading model from {config.pretrained_model}')
            state = torch.load(config.pretrained_model)
            model.load_state_dict(state['model_state_dict'], strict=True)

        return model

    @staticmethod
    def get_loss(config):

        loss_fn = get_loss(config.loss.name, config.loss.get('params'))

        return loss_fn

    @staticmethod
    def get_optimizer_and_scheduler(config, model):

        optimizer = torch.optim.Adam(model.parameters(), lr=config.solver.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=config.solver.patience, verbose=True)

        return optimizer, scheduler

    def display(self, pred, sample):
        # display a single image
        image = sample['image'][0]
        pred_boxes, _ = parse_detections(pred[0], threshold=0.1)
        gt_boxes = sample['boxes'][0]

        self.visualizer.display(image, pred_boxes.cpu(), gt_boxes)

    def forward(self, sample):
        images = sample['image'].to(self.device)
        heatmap = sample['heatmap'].to(self.device)
        scalemap = sample['scalemap'].to(self.device)
        classmap = sample['classmap'].to(self.device)

        pred = self.model(images)
        loss = self.loss_fn(pred, heatmap, scalemap, classmap)
        
        return pred, loss

    def train_step(self):
        config = self.config

        # define meters
        loss_meter = AverageMeter()
 
        self.model.train()
        for i, sample in enumerate(tqdm(self.train_dataset_it)):
            pred, loss = self.forward(sample)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item())

            if config.display and i % config.display_it == 0:
                with torch.no_grad():
                    self.display(pred, sample)

        return loss_meter.avg

    def val_step(self):
        config = self.config

        # define meters
        loss_meter = AverageMeter()
        ap_meter = APMeter()

        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tqdm(self.val_dataset_it)):
                pred, loss = self.forward(sample)
                loss_meter.update(loss.item())

                if config.display and i % config.display_it == 0:
                    self.display(pred, sample)

                # add detections to ap_meter
                pred_boxes, pred_scores = parse_batch_detections(pred, threshold=0.1)
                for pred, scores, gt in zip(pred_boxes, pred_scores, sample['boxes']):
                    ap_meter.add_detections(pred, scores, gt)

        # get ap metric
        ap, ap_dict = ap_meter.get_ap()
        metrics = {'map': ap, 'working_point': ap_dict}

        return loss_meter.avg, metrics

    def save_checkpoint(self, epoch, is_best_val=False, best_val_loss=0, is_best_ap=False, best_ap=0, metrics={}):
        config = self.config

        state = {
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
            "metrics": metrics
        }

        print("=> saving checkpoint")
        file_name = os.path.join(config.save_location, "checkpoint.pth")
        torch.save(state, file_name)
       
        if is_best_val:
            print("=> saving best_val checkpoint")
            shutil.copyfile(
                file_name, os.path.join(config.save_location, "best_val_model.pth")
            )

        if is_best_ap:
            print("=> saving best_ap checkpoint")
            shutil.copyfile(
                file_name, os.path.join(config.save_location, "best_ap_model.pth")
            )

    def train(self):  
        best_val_loss = float('inf')
        best_ap = 0

        # for epoch in range(config.solver.num_epochs):      
        epoch = 0
        while True:
            print(f'Starting epoch {epoch}')

            train_loss = self.train_step()
            val_loss, metrics = self.val_step()

            print(f'==> train loss: {train_loss}')
            print(f'==> val loss: {val_loss}')
            print(f'metrics: {metrics}')

            is_best_val = val_loss < best_val_loss
            best_val_loss = min(val_loss, best_val_loss)

            is_best_ap = metrics['map'] > best_ap
            best_ap = max(metrics['map'], best_ap)

            self.save_checkpoint(epoch, is_best_val=is_best_val, best_val_loss=best_val_loss, is_best_ap=is_best_ap, best_ap=best_ap, metrics=metrics)
            
            self.scheduler.step(val_loss)
            epoch = epoch + 1

            if self.optimizer.param_groups[0]['lr'] < self.config.solver.lr/100:
                break

            if epoch == self.config.solver.max_epochs:
                break
