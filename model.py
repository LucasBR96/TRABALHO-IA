import os
import sys
from tabnanny import verbose
from unittest import result

import torch
from torch import nn
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"
PATH = 'DATA/modelo.pt'

import numpy as np

CLASS_THRESHOLD = 0.5

class minha_net( nn.Module ):

    def __init__( self , threshold = CLASS_THRESHOLD ):

        super( minha_net, self).__init__()

        self.threshold = threshold
        self.seq = nn.Sequential(
            nn.Linear( 1100 , 20 ),
            nn.Sigmoid(),
            nn.Linear( 20 , 20 ),
            nn.Sigmoid(),
            nn.Linear( 20 , 20 ),
            nn.Sigmoid(),
            nn.Linear( 20 , 20 ),
            nn.Sigmoid(),
            nn.Linear( 20 , 20 ),
            nn.Sigmoid(),
            nn.Linear( 20 , 1 ),
            nn.Sigmoid()
            # nn.Threshold( self.threshold , 0 )
        )
    
    def forward( self , x ):
        result = self.seq( x )
        # result[ result >= self.threshold ] = 1
        # result[ result < self.threshold ] = 0
        return torch.squeeze( result )

def learning_step( X , y , mod : minha_net, opm : torch.optim.SGD , loss_fn ):
    
    y_prime = mod( X )
    train_loss = loss_fn( y_prime , y )
    opm.zero_grad()
    train_loss.backward()
    opm.step()

    return train_loss

def val_step( X_val , y_val , mod , loss_fn ):
    
    y_hat = mod( X_val )
    return loss_fn( y_hat , y_val ) 

def learning_loop( train_data , eval_data , lr = 1e-4 , batch_iters = 20 , num_batchs = 5000 , verbose = True ):

    mod = minha_net()
    loss_fn = nn.CrossEntropyLoss()

    opm = torch.optim.SGD( mod.parameters() , lr = lr )
    train_set = iter( DataLoader( train_data , batch_size = 50 , shuffle = True ) )
    test_set =  iter( DataLoader( eval_data , batch_size = 50 , shuffle = True ) )

    iters = []
    t_costs = []
    v_costs = []
    last_v_loss = sys.maxsize
    for i in range( num_batchs ):

        try:
            X , y = next( train_set )
        except StopIteration:
            train_set = iter( DataLoader( train_data , batch_size = 50 , shuffle = True ) )
            X , y = next( train_set )
        
        y_prime = mod( X )
        train_loss = loss_fn( y_prime , y )
        opm.zero_grad()
        train_loss.backward()
        opm.step()
        
        if i == 1 or i%batch_iters == 0:
            try:
                X_val , y_val = next( test_set )
            except StopIteration:
                test_set = iter( DataLoader( eval_data , batch_size = 50 , shuffle = True ) )
                X_val , y_val = next( test_set )
            
            with torch.no_grad():
                y_hat = mod( X_val )
                v_loss = loss_fn( y_hat , y_val )

                if v_loss < last_v_loss:
                    last_v_loss = v_loss
                    torch.save(mod.state_dict(), PATH)

            iters.append( i )
            t_costs.append( train_loss.item() )
            v_costs.append( v_loss.item() )

            if verbose:
                print( "-"*50 )
                print( f"iterac??o numero: {iters[ -1 ]}")
                print( f"custo no treino: {t_costs[ -1]:.2f}")
                print( f"custo na valida????o: {v_costs[ -1]:.2f}")
                print()

    
    return iters , t_costs , v_costs
