import torch
from torch import nn

KENELS_DEFAULT = [10,8,4,4,4]
STRIDES_DEFAULT = [5,4,2,2,2]
PADDINGS_DEFAULT = [0,0,0,0,0]
CHANNELS_DEFAULT = [512,512,512,512,512]

class CPC_LibriSpeech_Encoder(nn.Module):
    def __init__(self,
                 kernels = KENELS_DEFAULT,
                 channels = CHANNELS_DEFAULT,
                 strides = STRIDES_DEFAULT,
                 paddings = PADDINGS_DEFAULT,
                 lstm_features = 256,
                 tau = [1,2,4,8],
                 normalize = False
                ):
        
        super(CPC_LibriSpeech_Encoder,self).__init__()
        
        self.encoder = nn.Sequential(*[L for i,(k,c,s,p) in enumerate(zip(kernels,channels,strides,paddings)) for L in [nn.Conv1d(1 if i == 0 else channels[i-1], c, k, s, p),nn.ReLU()]])
        self.recurrent = nn.LSTM(channels[-1],lstm_features,batch_first = True)
        
        self.InfoNCE = batch_time_InfoNCE_loss(lstm_features,
                                               channels[-1],
                                               normalize = normalize,
                                               tau = tau)

    def encodings(self,X):
        #X - [batch, 1, time]
        
        Y = self.encoder(X).transpose(1,2)
        C,_ = self.recurrent(Y)
        return C,Y
    
    def forward(self,X):
        self.recurrent.flatten_parameters()
        C,Y = self.encodings(X)
        return self.InfoNCE(C,Y)
        
class batch_time_InfoNCE_loss(nn.Module):
    def __init__(self,context_size, encoding_size, normalize = False, tau = [1]):
        super(batch_time_InfoNCE_loss,self).__init__()

        self.InfoNCE = nn.ModuleList([multi_batch_InfoNCE_loss(context_size,encoding_size, normalize) for t in tau])
        self.tau = tau

    def forward(self,C,X):
        ###these both have shape [batch, time, features]
        ###I need to compute 
        return [I(C[:,:-t], X[:,t:]) for I,t in zip(self.InfoNCE,self.tau)]
        
class multi_batch_InfoNCE_loss(nn.Module):
    def __init__(self,context_size, encoding_size, normalize = False):
        super(multi_batch_InfoNCE_loss,self).__init__()

        self.transform = nn.Linear(encoding_size, context_size)
        self.normalize = normalize
        
    def forward(self,C,X):
        '''
        C - [batch, T, fc]
        X - [batch, T, fx]
        '''
        #I need to compute C.T.X and C.T.X~ where X~ means permuted across all bathces.
        #If I make a tensor which is dim. Z = [batch, batch, time - dt] where the (i,j) comp. is the combination of the (i,j) batch then I can compute:
        # sum_i[Z_ii - log(exp(sum(Z_i)))]
    
        Xt = self.transform(X)#[batch, fc]

        Z = torch.sum(C.unsqueeze(1)*Xt.unsqueeze(0),-1)#Z - [batch,batch]
        
        if self.normalize:
            Z = Z / (EPS + torch.sqrt(torch.sum(Z**2,-1,keepdims = True)))

        Zmax,_ = torch.max(Z,1,keepdims = True)
        Zdiag = torch.diagonal(Z).transpose(0,1)
        OUT = Zdiag - (torch.log(torch.sum(torch.exp(Z - Zmax),1)) + Zmax[:,0])   

        return torch.mean(OUT)
    
