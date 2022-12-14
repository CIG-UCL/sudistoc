import os
import glob
import random
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils as KU
import voxelmorph
import sudistoc
import matplotlib.pyplot as plt

import sys
print(sys.executable)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#%%

model_name = 'semi_unsupervised'
dataset = 'my_dataset'
model_dir = 'models/' + model_name

image1_files = sorted(glob.glob("datasets/" + dataset + "/ap/*.nii.gz")) 
image2_files = sorted(glob.glob("datasets/" + dataset + "/pa/*.nii.gz")) 
imageGT_files = None # sorted(glob.glob("datasets/" + dataset + "/unwarp/*.nii.gz"))
fieldGT_files = sorted(glob.glob("datasets/" + dataset + "/field/*.nii.gz"))
maskGT_files = sorted(glob.glob("datasets/" + dataset + "/weight/*.nii.gz"))

# some model parameters
unsup = True                        # switch to compute the unsupervised loss
unsup_isweighted = True
jacob_mod = True
transfo_constraint = 'oppsym'
int_steps = 0
int_downsize = 1
mask = maskGT_files is not None

# size of the training, validation and testing samples
n = len(image1_files)
n_train = np.floor(0.85*n)
n_val = n - n_train
batch_size = 1
ped = 1

# unet architecture
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]
n_downsamplings = len(enc_nf)

epochs = 1000
lr = 1e-4
epsilon = 0.00001

unsup_loss_func = sudistoc.losses.wMSE().loss 
sup_loss_func = sudistoc.losses.wMSE().loss 
reg_loss_func = voxelmorph.losses.Grad('l2', loss_mult=int_downsize).loss
img_weight = 200000
field_weight = 300
reg_weight = 1

#%%

kissing = image2_files is not None
print('Model recap:')
sudistoc.utils.model_recap(is_kissing=image2_files is not None, is_unsup=unsup, is_imageGT=imageGT_files is not None, is_fieldGT=fieldGT_files is not None, transfo_constraint=transfo_constraint, int_steps=int_steps, ped=ped, jacob_mod=jacob_mod, n_train=n_train, n_val=n_val, batch_size=batch_size)

ind = list(range(n))
np.random.shuffle(ind)

image1_files_train = [image1_files[i] for i in ind[0:n_train]] if image1_files is not None else None
image2_files_train = [image2_files[i] for i in ind[0:n_train]] if image2_files is not None else None
imageGT_files_train = [imageGT_files[i] for i in ind[0:n_train]] if imageGT_files  is not None else None
fieldGT_files_train = [fieldGT_files[i] for i in ind[0:n_train]] if fieldGT_files  is not None else None
maskGT_files_train = [maskGT_files[i] for i in ind[0:n_train]] if maskGT_files is not None else None

image1_files_val = [image1_files[i] for i in ind[n_train:n_train+n_val]] if image1_files is not None else None
image2_files_val = [image2_files[i] for i in ind[n_train:n_train+n_val]] if image2_files is not None else None
imageGT_files_val = [imageGT_files[i] for i in ind[n_train:n_train+n_val]] if imageGT_files is not None else None
fieldGT_files_val = [fieldGT_files[i] for i in ind[n_train:n_train+n_val]] if fieldGT_files is not None else None
maskGT_files_val = [maskGT_files[i] for i in ind[n_train:n_train+n_val]] if maskGT_files is not None else None


gen_train = sudistoc.generators.sudistoc_gen(image1_files=image1_files_train,
                                             image2_files=image2_files_train,
                                             imageGT_files=imageGT_files_train,
                                             fieldGT_files=fieldGT_files_train,
                                             weightGT_files=maskGT_files_train,
                                             ped=ped,
                                             unsup=unsup,
                                             unsup_isweighted=unsup_isweighted,
                                             n_downsamplings = n_downsamplings,
                                             batch_size=batch_size)

gen_val = sudistoc.generators.sudistoc_gen(image1_files=image1_files_val,
                                           image2_files=image2_files_val,
                                           imageGT_files=imageGT_files_val,
                                           fieldGT_files=fieldGT_files_val,
                                           weightGT_files=maskGT_files_val,
                                           ped=ped,
                                           unsup=unsup,
                                           unsup_isweighted=unsup_isweighted,
                                           n_downsamplings = n_downsamplings,
                                           batch_size=batch_size)


sample_train = next(gen_train)
weights = []
if maskGT_files is not None:
    for i in range(len(sample_train[1]) - 1):
        if i==0 and unsup and unsup_isweighted==False:
            continue
        else:
            print(i)
            weights += [sample_train[1][i][...,1][..., np.newaxis]]
            print(len(weights))
sample_train += (weights,)
sudistoc.utils.develop(sample_train)

inshape = sample_train[0][0].shape[1:-1]
nfeats = sample_train[0][0].shape[-1]
dims = len(inshape)

slicepos = 0.47

print('Input and groundtruth shapes')
sudistoc.utils.plot_input_GT(sample=sample_train, slicepos=slicepos)
plt.show()

#%%

# tensorflow device handling
device, nb_devices = voxelmorph.tf.utils.setup_device(1)
assert np.mod(batch_size, nb_devices) == 0, \
    'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (batch_size, nb_devices)


# build the model
model = sudistoc.networks.sudistoc_net(inshape=inshape,
                                       nb_unet_features=[enc_nf, dec_nf],
                                       int_steps=int_steps,
                                       int_downsize=int_downsize,
                                       src_feats=nfeats,
                                       trg_feats=nfeats,
                                       kissing=kissing,
                                       constraint=transfo_constraint,
                                       ped=ped,
                                       jacob_mod=jacob_mod,
                                       transfo_sup=fieldGT_files is not None,
                                       image_sup=imageGT_files is not None,
                                       unsup=unsup,
                                       name='disto_dense')

losses = []
weights = []
if unsup:
    losses += [unsup_loss_func]
    weights += [img_weight]
if imageGT_files is not None:
    losses += [sup_loss_func]
    if image2_files is not None:
        losses += [sup_loss_func]
        weights += [img_weight/2, img_weight/2]
    else:
        weights += [img_weight]
if fieldGT_files is not None:
    losses += [sup_loss_func]
    weights += [field_weight]
    if image2_files is not None:
        losses += [sup_loss_func]
        weights += [field_weight]
        
losses += [reg_loss_func]
weights += [reg_weight] 
weights = [float(weights[i]/sum(weights)) for i in range(len(weights))]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon), loss=losses, loss_weights=weights)

model.summary(line_length=130); print(' ')
KU.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, show_dtype=True, expand_nested=True,) 

print('Model parameters:')
sudistoc.utils.model_params(is_kissing=image2_files is not None, is_unsup=unsup, is_imageGT=imageGT_files is not None, is_fieldGT=fieldGT_files is not None, losses=losses.copy(), weights=weights.copy(), lr=lr, enc_nf=enc_nf, dec_nf=dec_nf)

trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
print('\nNumber of trainable parameters: ',trainableParams)

#%%

initial_epoch = 0
steps_per_epoch = n_train // batch_size
val_steps = n_val // batch_size

os.makedirs(model_dir, exist_ok=True)

save_filename = os.path.join(model_dir, model_name + '.h5')
model.save(save_filename.format(epoch=initial_epoch))
save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, monitor='val_loss', mode='min', save_best_only=True)

hist = model.fit(gen_train,
                 validation_data=gen_val,
                 validation_steps=val_steps,
                 initial_epoch=initial_epoch,
                 epochs=epochs,
                 steps_per_epoch=steps_per_epoch,
                 callbacks=[save_callback],
                 verbose=1)

#%%

loss_keys = list(hist.history.keys())
for i in range(int(len(loss_keys)/2)):
    sudistoc.utils.plot_history(hist, 0, [loss_keys[i], loss_keys[i+int(len(loss_keys)/2)]], ['b', 'r'], ['training', 'validation'])
    plt.show()

