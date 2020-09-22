from CPCLibriSpeech.model_management import build_models
from CPCLibriSpeech.data_management import get_data

import matplotlib.pyplot as plt
import glob
import pickle as pkl

import numpy as np

from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

import torch
from tqdm import tqdm
import json
import time
from options import opt
import os
import copy
import sys

test_dev = opt["test_dev"]
batch_size = opt["test_batch_size"]
num_workers = opt["num_workers"]

model_dir = str(sys.argv[1])

model = build_models.CPC_LibriSpeech_Encoder()
model.load_state_dict(torch.load(model_dir + "best_model_params",map_location = "cpu"))
model = model.to(test_dev)

test_speakers = json.load(open(model_dir + "/test_speakers.txt"))
test_p = [s for S in test_speakers for s in glob.glob(S + "**/*.flac")]

dataset = get_data.LibriSpeechDataset(test_p)
test_dataset = torch.utils.data.DataLoader(dataset,batch_size = batch_size,num_workers = num_workers,shuffle = True,drop_last = False)

try:
    c_data = pkl.load(open(model_dir + "c_data.pkl","rb"))
    e_data = pkl.load(open(model_dir + "e_data.pkl","rb"))
    m_data = pkl.load(open(model_dir + "m_data.pkl","rb"))
except:
    c_data = []
    e_data = []
    m_data = []
    with torch.no_grad():
        for B, spk, rec, sess, chk in tqdm(test_dataset):
            B = B.float().to(test_dev)
            C,E = model.encodings(B)
            spk = copy.deepcopy(spk)
            for sp,c,e in zip(spk.to("cpu").detach().numpy(),C.to("cpu").detach().numpy(),E.to("cpu").detach().numpy()):
                c_data.append(c[-1])
                e_data.append(e[-1])
                m_data.append(sp)
            del B
            del spk
            del chk
    c_data = np.array(c_data)
    e_data = np.array(e_data)
            
    pkl.dump(c_data,open(model_dir + "c_data.pkl","wb"))
    pkl.dump(e_data,open(model_dir + "e_data.pkl","wb"))
    pkl.dump(m_data,open(model_dir + "m_data.pkl","wb"))
    
data_len = len(c_data)
test_frac = int(.1*data_len)

c_data = np.array(c_data)
e_data = np.array(e_data)
lr_target = np.array(m_data)

c_lr = LogisticRegression()
c_lr.fit(c_data[:test_frac],lr_target[:test_frac])
e_lr = LogisticRegression()
e_lr.fit(e_data[:test_frac],lr_target[:test_frac])

c_train_score = c_lr.score(c_data[:test_frac],lr_target[:test_frac])
c_test_score = c_lr.score(c_data[test_frac:],lr_target[test_frac:])
e_train_score = e_lr.score(e_data[:test_frac],lr_target[:test_frac])
e_test_score = e_lr.score(e_data[test_frac:],lr_target[test_frac:])

print(f"LogisticRegression Result -\nRNN:\ttest: {np.round(c_test_score,3)*100}%\n\ttrain: {np.round(c_train_score,3)*100}%\nFFW:\ttest: {np.round(e_test_score,3)*100}%\n\ttrain: {np.round(e_train_score,3)*100}%")
json.dump({"RNN":{"test":c_test_score,"train":c_train_score},"FFW":{"test":e_test_score,"train":e_train_score}},open(model_dir + "LR_scores.json","w"))

c_tsne = TSNE(2)
e_tsne = TSNE(2)

tsne_frac = .1

tsne_spk = set(np.random.permutation(list(set(lr_target)))[:int(.1*len(set(lr_target)))])

print(f"Computing t-SNE for {len(tsne_spk)} speakers.")

tsne_ii = np.array([i for i,j in enumerate(lr_target) if j in tsne_spk])

c_embedding = c_tsne.fit_transform(c_data[tsne_ii])
e_embedding = e_tsne.fit_transform(e_data[tsne_ii])
tsne_targ = lr_target[tsne_ii]

fig,ax = plt.subplots(2,1,figsize = (5,10))

for c in set(tsne_targ):
    ii = np.where(tsne_targ == c)[0]
    cc = c_embedding[ii]
    ee = e_embedding[ii]
    ax[0].scatter(cc[:,0],cc[:,1],s = .01,alpha = 1)
    ax[1].scatter(ee[:,0],ee[:,1],s = .01,alpha = 1)
ax[0].set_title("RNN t-sne")
ax[1].set_title("FFW t-sne")
    
plt.savefig(model_dir + "tsne_embedding.pdf",bbox_inches = "tight")
plt.show()