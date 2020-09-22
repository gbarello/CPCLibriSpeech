from CPCLibriSpeech.model_management import build_models
from CPCLibriSpeech.data_management import get_data
import torch
from tqdm import tqdm
import json
import time
from options import opt
import os

dev = opt["dev"]
lr_step_rate = opt["lr_step_rate"]#10
root_data_path = opt["root_data_path"]#"/home/gabriel/Data/LibriSpeech/LibriSpeech_360/train-clean-360/"
dev_list = opt["dev_list"]
batch_size = opt["batch_size"]
num_workers = opt["num_workers"]
init_learning_rate = opt["init_learning_rate"]
lr_step_factor = opt["lr_step_factor"]
n_epochs = opt["n_epochs"]

od = f"./models/{int(time.time()*1000)}"
os.mkdir(od)

model = build_models.CPC_LibriSpeech_Encoder()
DP_model = torch.nn.DataParallel(model,dev_list,dev).to(dev)

(train_p,train_s),(test_p,test_s) = get_data.get_train_test_split(root_data_path)

json.dump(train_s,open(od + "/train_speakers.txt","w"))
json.dump(test_s,open(od + "/test_speakers.txt","w"))

train_dataset = torch.utils.data.DataLoader(train_p,batch_size = batch_size,num_workers = num_workers,shuffle = True)
test_dataset = torch.utils.data.DataLoader(test_p,batch_size = batch_size,num_workers = num_workers)

optimizer = torch.optim.Adam(model.parameters(),lr = init_learning_rate)
lr_step = torch.optim.lr_scheduler.MultiplicativeLR(optimizer,lambda epoch: lr_step_factor)

best_loss = None
for epoch in range(n_epochs):
    print(f"Epoch {epoch}")
    for phase in ["train","test"]:
        print(phase)
        if phase is "train":
            dataset = train_dataset
            DP_model.train()
        if phase is "test":
            dataset = test_dataset
            DP_model.eval()
            
        running_loss = 0
        for B,meta in tqdm(dataset):
            B.to(dev)
            
            loss = torch.stack(DP_model(B))
            loss = -torch.mean(loss)
            
            if phase is "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            running_loss += loss.item()
            
        running_loss /= len(dataset)
        
        print(f"{phase} loss: {running_loss}")
        if phase is "test" and (best_loss is None or running_loss < best_loss):
            torch.save(model.state_dict(),od + f"/best_model_params")
            
    if (epoch+1) % lr_step_rate == 0:
        lr_step.step()

torch.save(model.state_dict(),od + f"/model_params_final")
