# internal python imports
import os

# third party imports
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def develop(x, dec='|_'):
    if isinstance(x, (list, tuple)):
        print(dec, type(x), len(x))
        dec = '|    ' + dec
        for i in range(len(x)):
            develop(x[i], dec)
    elif isinstance(x, np.ndarray) or tf.is_tensor(x):
        print(dec, type(x), x.shape)   
    else: 
        print(dec, type(x))   


def plot_input_GT(sample, slicepos=0.47, rowtype=['Input', 'Ground truth', 'Weight']):
    
    slicey = 0
    pltcol = np.max((len(sample[0]),len(sample[1])))
    pltrow = len(sample)
    
    inshape = sample[0][0].shape[1:-1]
    dims = len(inshape)
    numslice = int(slicepos*inshape[2])
    
    dims=len(sample[0][0].shape[1:-1])
    
    for j in range(pltrow):
        print('  - ' + rowtype[j] + 's:')
        for i in range(len(sample[j])):
            plt.subplot(pltrow,pltcol,j*pltcol+i+1);
            if sample[j][i] is None :
                print('    ' + 'None')
                slicey = np.zeros_like(slicey)
            else:
                print('    ' + str(sample[j][i].shape))
                if dims == 2:
                    slicey = np.squeeze(sample[j][i][0,:,:,0])
                elif dims == 3:
                    slicey = np.squeeze(sample[j][i][0,:,:,numslice,0])
            plt.imshow(slicey) 
            ax = plt.gca();
            ax.grid(linestyle='-', linewidth=0.5)
            ax.set_xticklabels([])
            ax.set_xticks(range(0, slicey.shape[1],10))
            ax.set_yticklabels([])
            ax.set_yticks(range(0, slicey.shape[0],10))
            ax.tick_params(direction='out', length=0, width=0, grid_color='grey', grid_alpha=0.75)
            ax.invert_xaxis()
            ax.invert_yaxis()
            plt.title(rowtype[j] + ' ' + str(i), fontsize=10)
            

def plot_img(sample, slicepos=0.47, colnames=None, cmin=[None], cmax=[None]):

    slicey=0
    pltcol = len(sample)
    
    if len(cmin) == 1: cmin = cmin * pltcol 
    if len(cmax) == 1: cmax = cmax * pltcol 
    
    inshape = sample[0].shape[1:-1]
    dims = len(inshape)
    numslice = int(slicepos*inshape[2])
    
    dims=len(sample[0].shape[1:-1])

    for i in range(len(sample)):
        if sample[i].shape[0] == 1: sample[i] = sample[i][0, ...]
        plt.subplot(1,pltcol,i+1);
        if dims == 2:
            slicey = np.squeeze(sample[i][:,:,0])
        elif dims == 3:
            slicey = np.squeeze(sample[i][:,:,numslice,0])
            
        plt.imshow(slicey, vmin=cmin[i], vmax=cmax[i])   
        ax = plt.gca();
        ax.grid(linestyle='-', linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_xticks(range(0,slicey.shape[1],10))
        ax.set_yticklabels([])
        ax.set_yticks(range(0,slicey.shape[0],10))
        ax.tick_params(direction='out', length=0, width=0, grid_color='grey', grid_alpha=0.75)
        ax.invert_xaxis()
        ax.invert_yaxis()
        plt.title(colnames[i], fontsize=10)
            
            
def model_recap(is_kissing, is_unsup, is_imageGT, is_fieldGT, transfo_constraint, int_steps, ped, jacob_mod, n_train, n_val, batch_size):
    
    if is_kissing:
        print('  - 2 analogous input images.')
    else: 
        print('  - 1 input image.')
    if ped is not None:
        print('  - Unidirectional registration, constrained along axis ' + str(ped) + '.')
    else: 
        print('  - Unconstrained registration (all directions).')
    if transfo_constraint == 'oppsym':
        print('  - Opposite symmetry constraint on transformations.')
    elif transfo_constraint == 'diffeo':
        print('  - Inversion symmetry constraint on transformations.')
    print('  - Number of integration steps: ' + str(int_steps) + '.')
    if jacob_mod:
        print('  - Jacobian determinant intensity modulation.')
    if is_unsup:
        print('  - Unsupervised similarity between moved images.')
    if is_imageGT:
        print('  - Supervision by similarity to ground truth image.')
    if is_fieldGT:
        print('  - Supervision by similarity to ground truth field.')
    print('  - Training sample size: ' + str(n_train) + '.')
    print('  - Validation sample size: ' + str(n_val) + '.')
    print('  - Batch size: ' + str(batch_size) + '.')
    print(' ')


def model_params(is_kissing, is_unsup, is_imageGT, is_fieldGT, losses, weights, lr, enc_nf, dec_nf):
    
    if is_unsup:
        loss = losses.pop(0)
        weight = weights.pop(0)
        print('  - Unsupervised loss: ' + str(loss.__self__.__class__.__name__) + ', w=' + str(weight) + '.')
    if is_imageGT:
        loss = losses.pop(0)
        weight = weights.pop(0)
        print('  - Supervised loss on image 1: ' + str(loss.__self__.__class__.__name__) + ', w=' + str(weight) + '.')
        if is_kissing:
            loss = losses.pop(0)
            weight = weights.pop(0)
            print('  - Supervised loss on image 2: ' + str(loss.__self__.__class__.__name__) + ', w=' + str(weight) + '.')
    if is_fieldGT:
        loss = losses.pop(0)
        weight = weights.pop(0)
        print('  - Supervised loss on field 1: ' + str(loss.__self__.__class__.__name__) + ', w=' + str(weight) + '.')
        if is_kissing:
            loss = losses.pop(0)
            weight = weights.pop(0)
            print('  - Supervised loss on field 2: ' + str(loss.__self__.__class__.__name__) + ', w=' + str(weight) + '.')
    loss = losses.pop(0)
    weight = weights.pop(0)    
    print('  - Regularization loss: ' + str(loss.__self__.__class__.__name__) + ', w=' + str(weight) + '.')
    print('  - Learning rate: ' + str(lr) + '.')
    print('  - Encoder: ' + str(enc_nf) + '.')
    print('  - Decoder: ' + str(dec_nf) + '.')  
    
    
def shift_to_transfo(loc_shift, indexing='ij'):
    
    if isinstance(loc_shift.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = loc_shift.shape[1:-1].as_list()
    else:
        volshape = loc_shift.shape[1:-1]
    ndims = len(volshape)
    
    ij = [range(volshape[i]) for i in range(ndims)]
    mesh = tf.meshgrid(*ij, indexing=indexing)
    mesh = tf.cast(tf.expand_dims(tf.stack(mesh, axis=-1), 0), 'float32')
    
    return mesh + loc_shift

    
def plot_history(hist, start=0, loss_name=['loss'], color=None, labels=None):

    if color is None: color = ['b']*len(loss_name)
    ymax=-1000
    ymin=1000
    for i, loss in enumerate(loss_name):
      maxi = max(hist.history[loss][start:])
      mini = min(hist.history[loss][start:])
      plt.plot(hist.epoch[start:], hist.history[loss][start:], '-', color=color[i])
      ymax=max((ymax,maxi))
      ymin=min((ymin,mini))
    plt.xlabel('epoch')
    plt.ylim((ymin,ymax))
    if labels is not None:
        plt.legend(labels)
    plt.title(loss_name[0])
   # plt.show()


     
def jacobian(transfo, outDet=False):

    if isinstance(transfo.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
        volshape = transfo.shape[1:-1].as_list()
    else:
        volshape = transfo.shape[1:-1]
    ndims = len(volshape)
    
    jacob = [None] * ndims
    for d in range(ndims):
        grad = tf.gather(transfo, range(2, volshape[d]), axis=d+1)-tf.gather(transfo, range(volshape[d]-2), axis=d+1)
        grad = tf.expand_dims(grad, axis=ndims+1)
        grad_left = tf.gather(transfo, 1, axis=d+1)-tf.gather(transfo, 0, axis=d+1)
        grad_left = tf.expand_dims(tf.expand_dims(grad_left, d+1), ndims+1)
        grad_right = tf.gather(transfo, volshape[d]-1, axis=d+1)-tf.gather(transfo, volshape[d]-2, axis=d+1)
        grad_right = tf.expand_dims(tf.expand_dims(grad_right, d+1), ndims+1)
        jacob[d] = tf.concat((grad_left, grad/2, grad_right), axis=d+1)
    
    jacob = tf.concat(jacob, axis=ndims+1) 
       
    if outDet:
        # detjac = tf.linalg.det(jacob)
        if ndims == 2:
            detjac =  jacob[:,:,:,0,0] * jacob[:,:,:,1,1]\
                    - jacob[:,:,:,1,0] * jacob[:,:,:,0,1] 
        elif ndims == 3:
            detjac =  jacob[:,:,:,:,0,0] * jacob[:,:,:,:,1,1] * jacob[:,:,:,:,2,2]\
                    + jacob[:,:,:,:,0,1] * jacob[:,:,:,:,1,2] * jacob[:,:,:,:,2,0]\
                    + jacob[:,:,:,:,0,2] * jacob[:,:,:,:,1,0] * jacob[:,:,:,:,2,1]\
                    - jacob[:,:,:,:,2,0] * jacob[:,:,:,:,1,1] * jacob[:,:,:,:,0,2]\
                    - jacob[:,:,:,:,1,0] * jacob[:,:,:,:,0,1] * jacob[:,:,:,:,2,2]\
                    - jacob[:,:,:,:,0,0] * jacob[:,:,:,:,2,1] * jacob[:,:,:,:,1,2]
        else:
            raise Exception('Only dimension 2 or 3 supported, but got: %s' % ndims)
            
        return jacob, detjac
    else: 
        return jacob 

