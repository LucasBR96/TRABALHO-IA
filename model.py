import os

import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np

class minha_net( nn.Module ):

    def __init__( self ):

        super( minha_net, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear( 1100 , 20 ),
            nn.ReLU(),
            nn.Linear( 20 , 20 ),
            nn.ReLU(),
            nn.Linear( 20 , 20 ),
            nn.ReLU(),
            nn.Linear( 20 , 20 ),
            nn.ReLU(),
            nn.Linear( 20 , 20 ),
            nn.ReLU(),
            nn.Linear( 20 , 1 ),
            nn.Sigmoid(),
        )
    
    def foward( self , x ):
        return self.seq( x )


def learning_step( X , y , mod : minha_net, opm : torch.optim.SGD , loss_fn ):
    
    y_prime = mod( X )
    train_loss = loss_fn( y , y_prime )
    opm.zero_grad( )
    train_loss.backward()
    opm.step()

    return train_loss

def val_step( X_val , y_val , mod , loss_fn ):
    
    y_hat = mod( X_val )
    return loss_fn( y_val , y_hat ) 

m = minha_net()
for name, param in m.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")