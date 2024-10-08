from ctypes import Union
from typing import Any
import torch
import os
import numpy as np
from torch.nn import CTCLoss

#from base.base_trainer import BaseTrainer
from tqdm import tqdm
#from torch.cuda.amp import autocast
#from logger.pbar import PBar
from typing import Dict, Union
from utils.progress_bar import Pro_BAR
from utils.tensorboard import TensorboardWriter


class Train():
    def __init__(self, 
                #dist,
                #rank,
                #n_gpus,
                device,
                structure,
                epochs,
                steps_per_epoch,
                model,
                metric_wer,
                processor,
                train_dl,
                valid_dl,
                #train_sampler,
                #val_sampler,
                optimizer,
                save_dir,
                log_dir,
                grad_accm_steps,
                #use_amp,
                #max_clip_grad_norm
                ):
        self.device = device
        self.structure = structure
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.model = model
        self.metric_wer = metric_wer
        self.processor = processor
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.grad_accm_steps = grad_accm_steps
        #self.n_gpus = n_gpus
        #self.max_clip_grad_norm = max_clip_grad_norm
        self.sr = self.structure["main"]["sr"]
        self.stateful_metrics = ["train_loss", "train_lr", "train_wer", "val_loss", "val_wer"]
        self.start_epoch = 0
        self.probar_step = 0
        self.use_distill = False
        self.completed_steps = 0


        self.valid_interval = self.structure["training"]["args"]["validation_interval"]
        self.save_max_metric_score = self.structure["training"]["args"]["save_max_metric_score"]
        self.best_score = -np.inf if self.save_max_metric_score else np.inf

        self.processor.save_pretrained(structure["pretrained_model"]["path"])

        self.writer = TensorboardWriter(self.log_dir)
        self._count_parameters()
        self._count_trainable_parameters()
        self.loss_func = CTCLoss(reduction='mean')
        #self.total_loss = 0

    def _count_parameters(self):
        params_of_network = 0
        for param in self.model.parameters():
            params_of_network += param.numel()
        print(f"The amount of parameters in the project is {params_of_network / 1e6} million.")

    def _count_trainable_parameters(self) -> None:
        print("Number of trainable params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            # self.model.train()
            self._train_epoch(epoch)

    def _train_epoch(self,epoch):
        print("Epoch {}: ".format(epoch+1))
        probar = Pro_BAR(self.steps_per_epoch, 10, stateful_metrics = self.stateful_metrics)
        

        for dl_step, batch in enumerate(self.train_dl):
            #print(torch.cuda.max_memory_allocated())            
            #input_values = batch['input_values'].to(self.device)
            #attention_mask = batch['attention_mask'].to(self.device)
            #input_ids = batch['labels'].to(self.device)
            batch.to(self.device)
            #print(f'input values shape {input_values.shape}')
            #a=input("enter anything?")
            #self.model.train()
            outputs = self.model(**batch)
            #print(outputs)
            #print(f"output:{outputs.loss}")
            #loss = self.loss_func(outputs,batch['labels'])
            #self.total_loss += (loss)/(self.grad_accm_steps)
            loss = (outputs.loss)/(self.grad_accm_steps)
            #print('losss completed')
            loss.backward()
            


            if (dl_step+1) % self.grad_accm_steps == 0:
                #self.total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                outputs = self.model(**batch)
                #wer = torch.tensor(self.metric_wer(outputs.logits.detach(),batch['labels']),dtype=float)
                wer = self.metric_wer(outputs.logits.detach(),batch['labels'])
                train_logs = {
                    "train_loss" :loss,
                    "train_lr" : self.optimizer.param_groups[-1]['lr'],
                    "train_wer" : wer
                }
                train_logs = {k: v.item() if hasattr(v, 'item') else v for k, v in train_logs.items()}
                self.writer.update(self.completed_steps, 'Train', train_logs)
                probar.update(self.probar_step+1, train_logs)

               # self.total_loss = 0

                if (self.completed_steps+1) % self.valid_interval == 0:
                    
                    print("\nValidation is in progress...")
                    self.model.eval()
                    val_logs = self._valid_epoch(self.completed_steps)
                        
                    self.writer.update(self.completed_steps, 'Validation', val_logs)
                    probar.update(self.probar_step+1, val_logs)

                    if self._is_best_epoch(val_logs['val_wer'], save_max_metric_score=self.save_max_metric_score):
                        self._save_checkpoint(epoch, dl_step, is_best_epoch=True)
                    else:
                        self._save_checkpoint(epoch, dl_step, is_best_epoch=False)
                self.probar_step += 1
                self.completed_steps += 1
        torch.cuda.empty_cache()


    def _valid_epoch(self, step):
        val_logs = {
            "val_loss": 0,
            "val_wer": 0
        }

        for batch in tqdm(self.valid_dl, total = len(self.valid_dl)):
            batch.to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch)

            val_logs["val_loss"] += outputs.loss / len(self.valid_dl)
            val_logs["val_wer"] += torch.tensor(self.metric_wer(outputs.logits, batch['labels'])) / len(self.valid_dl)

        val_logs = {k: v.item() if hasattr(v, 'item') else v for k, v in val_logs.items()}
        return val_logs


    def _is_best_epoch(self, score, save_max_metric_score=True):
        
        if save_max_metric_score and score >= self.best_score:
            self.best_score = score
            return True
        elif not save_max_metric_score and score <= self.best_score:
            self.best_score = score
            return True
        else:
            return False



    def _save_checkpoint(self, epoch: int, dl_step: int, is_best_epoch: bool = False) -> None:
        print(f"\n Saving model checkpoint...")

        state_dict = {
            "epoch": epoch,
            "dl_step": dl_step,
            "probar_step": self.probar_step,
            "best_score": self.best_score,
            "optimizer": self.optimizer.state_dict(),
            "completed_steps": self.completed_steps
        }

        state_dict["model"] = self.model.state_dict()

        torch.save(state_dict, os.path.join(self.save_dir, "latest_model.tar"))
        torch.save(state_dict, os.path.join(self.save_dir, f"model_{str(self.completed_steps+1)}.tar"))
        if is_best_epoch:
            torch.save(state_dict, os.path.join(self.save_dir, "best_model.tar"))
            self.model.save_pretrained(self.structure["pretrained_model"]["path"])


