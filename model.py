import os
import sys

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np

CLASS_THRESHOLD = 0.5

class minha_net( nn.Module ):

    def __init__( self , threshold = CLASS_THRESHOLD ):

        super( minha_net, self).__init__()

        self.threshold = threshold
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

        y_hat   = self.seq( x )
        y_prime = torch.zeros( len( y_hat ) )
        y_prime[ y_hat >= self.threshold ] = 1

        return y_prime

def learning_step( X , y , mod : minha_net, opm : torch.optim.SGD , loss_fn ):
    
    y_prime = mod( X )
    train_loss = loss_fn( y , y_prime )
    opm.zero_grad()
    train_loss.backward()
    opm.step()

    return train_loss

def val_step( X_val , y_val , mod , loss_fn ):
    
    y_hat = mod( X_val )
    return loss_fn( y_val , y_hat ) 

def bin_F1_score( y_hat , y ):

    TP = sum( y_hat & y )
    FP = sum( y_hat & ( 1 - y ) )
    # TN = sum( ( 1 - y_hat ) & ( 1 - y ) )
    FN = sum( ( 1 - y_hat ) & y )

    precision = TP/( TP + FP )
    recall    = TP/( TP + FN )

    return 2*precision*recall/( precision + recall )

#-------------------------------------------------------------------------------------------
# Full batch, not used in "production"
def learning_loop( X , y , X_val , y_val , lr = 1e-3 , batch_iters = 40 , num_batchs = 100  ):

    opm = torch.optim.SGD( lr = lr )
    def loss( y , y_hat ):
        return -100*bin_F1_score( y_hat , y )
    mod = minha_net()

    iters = []
    t_costs = []
    v_costs = []
    last_v_loss = sys.maxsize
    for i in range( num_batchs ):
        for j in range( batch_iters ):
            t_loss = learning_step( X , y , mod , opm , loss )
        v_loss = val_step( X_val , y_val , mod , loss )

        if v_loss < last_v_loss

        iters.append( ( i + 1 )*batch_iters )
        t_costs.append( t_loss )
        v_costs.append( v_loss )