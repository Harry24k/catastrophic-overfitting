import os

import torch
from torch.optim import *
from torch.optim.lr_scheduler import *

from torchhk import RecordManager    

r"""
Trainer.

Attributes:
    self.model : model.
    self.device : device where model is.
    self.optimizer : optimizer.
    self.scheduler : scheduler (* Automatically Updated).
    self.max_epoch : total number of epochs.
    self.max_iter : total number of iterations.
    self.epoch : current epoch.
        * Starts from 0.
        * Automatically updated.
    self.iter : current iter.
        * Starts from 0.
        * Automatically updated.
    self.record_keys : items to record (= items returned by do_iter).

"""

class Trainer():
    def __init__(self, name, model, **kwargs):
        # Set Model
        self.name = name
        self.model = model
        self.device = next(model.parameters()).device
        
        # Set Custom Arguments
        for k, v in kwargs.items():
#             assert( k in [])
            setattr(self, k, v)
        
    def train(self, train_loader, max_epoch=200, start_epoch=0,
              optimizer=None, scheduler=None, scheduler_type=None,
              save_type="Epoch", save_path=None, save_overwrite=False, 
              record_type="Epoch"):
        
        # Set Train Mode
        self.model.train()
        
        # Set Epoch and Iterations
        self.max_epoch = max_epoch
        self.max_iter = len(train_loader)
            
        # Set Optimizer and Schduler
        self._init_optim(optimizer, scheduler, scheduler_type)
        
        # Check Save and Record Values
        self._check_valid_options(save_type)
        self._check_valid_options(record_type)
        
        # Check Save Path is given
        if save_type is None:
            save_type = "None"
        else:
            # Check Save Path
            if save_path is not None:
                # Save Initial Model
                self._check_path(save_path, overwrite=save_overwrite)
                self._save_model(save_path, 0)
            else:
                raise ValueError("save_path should be given for save_type != None.")
            
        # Print Training Information
        if record_type is not None:
            self._init_record(record_type)
            print("["+self.name+"]")
            print("Training Information.")
            print("-Epochs:",self.max_epoch)
            print("-Optimizer:",self.optimizer)
            print("-Scheduler:",self.scheduler)
            print("-Save Path:",save_path)
            print("-Save Type:",save_type)
            print("-Record Type:",record_type)
            print("-Device:",self.device)
        
        # Training Start
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            epoch_record = []
            
            if epoch < start_epoch:
                if self.scheduler_type == "Epoch":
                    self._update_scheduler()
                elif self.scheduler_type == "Iter":
                    for i in range(max_iter):
                        self._update_scheduler()
                else:
                    pass
                continue
            
            for i, train_data in enumerate(train_loader):
                self.iter = i
                iter_record = self._do_iter(train_data)
                self.rm.progress()
                
                # Check Last Batch
                is_last_batch = (i+1==self.max_iter)
                    
                # Update Records
                if record_type == "Epoch":
                    epoch_record.append(iter_record)
                    if is_last_batch:
                        epoch_record = torch.tensor(epoch_record).float()
                        self._update_record([epoch+1,
                                            *[r.item() for r in epoch_record.mean(dim=0)]])
                        epoch_record = []
                elif record_type == "Iter":
                    self._update_record([epoch+1, i+1, *iter_record])
                else:
                    pass

                # Save Model
                if save_type == "Epoch":
                    if is_last_batch:
                        self._save_model(save_path, epoch+1)
                elif save_type == "Iter":
                    self._save_model(save_path, epoch+1, i+1)
                else:
                    pass
                
                # Scheduler Step
                if self.scheduler_type=="Epoch" and is_last_batch:
                    self._update_scheduler()
                elif self.scheduler_type=="Iter":
                    self._update_scheduler()
                else:
                    pass
                
                # Set Train Mode
                self.model = self.model.to(self.device)
                self.model.train()
                
        # Print Summary
        try:
            if record_type is not None:
                self.rm.summary()
        except Exception as e:
            print("Summary Error:",e)
        
    def save_all(self, save_path, overwrite=False):
        self._check_path(save_path+".pth", overwrite=overwrite, file=True)
        self._check_path(save_path+".csv", overwrite=overwrite, file=True)
        print("Saving Model")
        torch.save(self.model.cpu().state_dict(), save_path+".pth")
        print("...Saved as pth to %s !"%(save_path+".pth"))
        print("Saving Records")
        self.rm.to_csv(save_path+".csv")
        self.model.to(self.device)
    
    #############################
    # OVERRIDE BELOW FUNCTIONS #
    ############################
    
    # Do Iter
    def _do_iter(self, images, labels):
        raise NotImplementedError
        
    # Scheduler Update
    def _update_scheduler(self):
        self.scheduler.step()
        
    ####################################
    # DO NOT OVERRIDE BELOW FUNCTIONS #
    ###################################
            
    # Initialization RecordManager
    def _init_record(self, record_type):
        keys = ["Epoch"]
        if record_type == "Iter":
            keys = ["Epoch", "Iter"]
            
        for key in self.record_keys:
            keys.append(key)
            
        keys.append("lr")
        self.rm = RecordManager(keys)
    
    # Update Records
    def _update_record(self, records):
        self.rm.add([*records, self.optimizer.param_groups[0]['lr']])
        
    # Set Optimizer and Scheduler
    def _init_optim(self, optimizer, scheduler, scheduler_type):
        # Set Optimizer
        if not isinstance(optimizer, str):
            self.optimizer = optimizer     
        else:
            exec("self.optimizer = " + optimizer.split("(")[0] + "(self.model.parameters()," + optimizer.split("(")[1])

        # Set Scheduler
        if not isinstance(scheduler, str):
            self.scheduler = scheduler
            if self.scheduler is None:
                self.scheduler_type = None
            else:
                if scheduler_type is None:
                    raise ValueError("The type of scheduler must be specified as 'Epoch' or 'Iter'.")
                self.scheduler_type = scheduler_type
        else:
            if "Step(" in scheduler:
                # Step(milestones=[2, 4], gamma=0.1)
                exec("self.scheduler = " + "MultiStepLR(self.optimizer, " + scheduler.split("(")[1])
                self.scheduler_type = 'Epoch'

            elif 'Cyclic(' in scheduler:
                # Cyclic(base_lr=0, max_lr=0.3)
                lr_steps = self.max_epoch * self.max_iter
                exec("self.scheduler = " + "CyclicLR(self.optimizer, " + scheduler.split("(")[1].split(")")[0] + \
                     ", step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)")
                self.scheduler_type = 'Iter'

            elif 'Cosine' == scheduler:
                # Cosine
                self.scheduler = CosineAnnealingLR(self.optimizer, self.max_epoch, eta_min=0)
                self.scheduler_type = 'Epoch'
                
            else:
                exec("self.scheduler = " + scheduler.split("(")[0] + "(self.optimizer, " + scheduler.split("(")[1])
                self.scheduler_type = scheduler_type
            
    def _save_model(self, save_path, epoch, i=0):
        torch.save(self.model.cpu().state_dict(),
                   save_path+"/"+str(epoch).zfill(len(str(self.max_epoch)))\
                   +"_"+str(i).zfill(len(str(self.max_iter)))+".pth")
        self.model.to(self.device)
        
    # Check and Create Path
    def _check_path(self, path, overwrite=False, file=False):
        if os.path.exists(path):
            if overwrite:
                print("Warning: Save files will be overwritten!")
            else:
                raise ValueError('[%s] is already exists.'%(path))
        else:
            if not file:
                os.makedirs(path)
                
    # Check Valid Options
    def _check_valid_options(self, key):
        if key in ["Epoch", "Iter", None]:
            pass
        else:
            raise ValueError(key, " is not valid. [Hint:'Epoch', 'Iter', None]")