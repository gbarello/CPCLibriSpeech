import numpy as np                                                             
import soundfile as sf 
import torch
import glob
from tqdm import tqdm
from torchvision import transforms as trans
from functools import partial
from multiprocessing import Pool, cpu_count

def test_file(fn,chunk_len = 20480):
    return len(process_file(fn,chunk_len = chunk_len)[0])

def process_file(fn, chunk_len = 20480, drop_last = True):
    data, samplerate = sf.read(fn)
    
    data = [np.expand_dims(data[k:k+chunk_len],0) for k in range(0,len(data) - (chunk_len + 1 if drop_last  else 0),chunk_len)]
    
    return data, samplerate


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self,file_list,chunk_len = 20480):
        self.all_files = file_list        
        self.chunk_len = chunk_len
        
        plen = partial(test_file,chunk_len = chunk_len)
        p = Pool(cpu_count())
        self.lens = p.map(plen,self.all_files)
        p.close()
        p.join()
        
        self.indices = [(i,j) for i in range(len(self.all_files)) for j in range(self.lens[i])]
        
        self.data_by_index = [tuple(map(int,s.split("/")[-1].split(".")[0].split("-"))) for s in self.all_files]
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self,idx):
        file,chunk = self.indices[idx]
        record = process_file(self.all_files[file],chunk_len = self.chunk_len)[0][chunk]
        spk,rec,sess = self.data_by_index[file]
        return torch.from_numpy(record).float(),spk,rec,sess,chunk
        
def get_train_test_split(root_path, test_frac = .2, seed = 1984,chunk_len = 20480):
    np.random.seed(seed)
    
    all_speakers = np.random.permutation(glob.glob(root_path + "**/"))
    
    spk_len = [len(glob.glob(s + "**/*.flac")) for s in all_speakers]
    spk_frac = np.cumsum(spk_len)/np.sum(spk_len)
    
    test_id = np.where(spk_frac < test_frac)[0]
    train_id = np.where(spk_frac >= test_frac)[0]

    train_spk = [all_speakers[s] for s in train_id]
    test_spk = [all_speakers[s] for s in test_id]
    
    train_paths = [s for i in train_spk for s in glob.glob(i + "**/*.flac")]
    test_paths = [s for i in test_spk for s in glob.glob(i + "**/*.flac")]
 
    train_paths = LibriSpeechDataset(train_paths)
    test_paths = LibriSpeechDataset(test_paths)
    
    return (train_paths, train_spk), (test_paths, test_spk)
    