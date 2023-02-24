import sys
sys.path.append("utils/")
import yaml
import utils.dataset as dataset
import utils.trainer as trainer
from torch.utils.tensorboard import SummaryWriter
import torch

import models
import GloVe_loss


if __name__ == '__main__':
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
        
    
    model_name = config["model_name"]
    batch_size = config["batch_size"]
    number_of_epochs = config["epochs"]
    lr = config["learning_rate"]
    
    
    train_loader,vocab = dataset.get_dataloader_and_vocab(split="train",batch_size=batch_size,vocab=None)
    val_loader,_ = dataset.get_dataloader_and_vocab(split="test",batch_size=batch_size,vocab=vocab)
    model = models.GloVeModel(len(vocab))
    criterion = GloVe_loss.GloVeLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler=None
    # device = torch.device("mps" if torch.has_mps else "cpu")
    device = torch.device("cpu")
    model_path = f"weights/{model_name}"
    writer = SummaryWriter(f".runs/{model_name}")
    
    t = trainer.Trainer(model,train_loader,val_loader,number_of_epochs,criterion,optimizer,scheduler,device,model_path,model_name,writer)
    t.train()
