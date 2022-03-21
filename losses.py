import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.backend as K
from . import utils


class MSE:
    """
    Weighted mean squared error.
    """
    def loss(self, y_true, y_pred, sample_weight=None): 
        
        if sample_weight is None:
            sample_weight = K.ones_like(y_true)
        
        loss = K.sum(sample_weight * K.square(y_true - y_pred)) / K.sum(sample_weight)
        
        return loss
  
    
class Jacob:
    """
    N-D jacobian loss.
    functype: 'logarithmic' or 'rational'.
          In both case: f(x)=f(1/x) so contractions and expansions a treated the same.
    power: power.
    """
    
    def __init__(self, isshift=True, functype='logarithmic', power=2, epsilon=10e-3):
        self.isshift = True
        self.functype = functype
        self.power = power
        self.epsilon = epsilon
      
    def _jac(self, f):
        
        if self.isshift:
            _, jacf = jacobian(shift_to_transfo(f), outDet=True) 
        else:
            _, jacf = jacobian(f, outDet=True)
    
        return jacf

    def loss(self, _, y_pred):

        jacf = self._jac(y_pred)
        
        m_leps = K.less(jacf, self.epsilon)
        jacf_leps = jacf[m_leps]
        jacf_leps = K.exp(jacf_leps / self.epsilon + np.log(self.epsilon) - 1) + 10e-30;
        m_geps = K.greater(jacf, self.epsilon)  
        jacf_geps = jacf[m_geps]
        jacf = K.concatenate((jacf_leps, jacf_geps))          
        
        if self.functype == 'logarithmic':
            funcjac = K.log(jacf)
            loss = K.mean(K.pow(K.abs(funcjac), self.power))
            
        elif self.functype == 'rational':
            funcjac = (jacf + K.pow(jacf, -1)) / 2
            loss = K.mean(K.pow(K.abs(funcjac), self.power)-1)
  
        return loss
 