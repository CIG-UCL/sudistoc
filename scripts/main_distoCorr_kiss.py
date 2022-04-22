import os
import shutil
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

model_name='test'
dataset = 'possum'

# image1_files = sorted(glob.glob("datasets/b0/medium/*_AP.nii.gz"))
# image2_files = sorted(glob.glob("datasets/b0/medium/*_PA.nii.gz"))
# imageGT_files = None
# fieldGT_files = sorted(glob.glob("datasets/b0/medium/topupField*.nii.gz"))
# unsup = True

image1_files = sorted(glob.glob("datasets/" + dataset + "/*ap.nii.gz")) 
image2_files = sorted(glob.glob("datasets/" + dataset + "/*pa.nii.gz")) 
imageGT_files = None # sorted(glob.glob("datasets/" + dataset + "/*sim.nii.gz"))
fieldGT_files = sorted(glob.glob("datasets/" + dataset + "/*field.nii.gz"))
maskGT_files = sorted(glob.glob("datasets/" + dataset + "/*weight.nii.gz"))

unsup = True
jacob_mod = True
transfo_constraint = 'oppsym'
int_steps = 0
int_downsize = 1
mask = maskGT_files is not None

n = len(image1_files)
n_train = 20
n_val = 10
n_test = n - n_train - n_val
batch_size = 1
ped = 1

epochs = 150

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

image1_files_test = [image1_files[i] for i in ind[n_train+n_val:n_train+n_val+n_test]] if image1_files is not None else None
image2_files_test = [image2_files[i] for i in ind[n_train+n_val:n_train+n_val+n_test]] if image2_files is not None else None
imageGT_files_test = [imageGT_files[i] for i in ind[n_train+n_val:n_train+n_val+n_test]] if imageGT_files is not None else None
fieldGT_files_test = [fieldGT_files[i] for i in ind[n_train+n_val:n_train+n_val+n_test]] if fieldGT_files is not None else None
maskGT_files_test = [maskGT_files[i] for i in ind[n_train+n_val:n_train+n_val+n_test]] if maskGT_files is not None else None

gen_train = sudistoc.generators.sudistoc_gen(image1_files=image1_files_train,
                                             image2_files=image2_files_train,
                                             imageGT_files=imageGT_files_train,
                                             fieldGT_files=fieldGT_files_train,
                                             weightGT_files=maskGT_files_train,
                                             ped=ped,
                                             unsup=unsup,
                                             batch_size=batch_size)

gen_val = sudistoc.generators.sudistoc_gen(image1_files=image1_files_val,
                                           image2_files=image2_files_val,
                                           imageGT_files=imageGT_files_val,
                                           fieldGT_files=fieldGT_files_val,
                                           weightGT_files=maskGT_files_val,
                                           ped=ped,
                                           unsup=unsup,
                                           batch_size=batch_size)

gen_test = sudistoc.generators.sudistoc_gen(image1_files=image1_files_test,
                                            image2_files=image2_files_test,
                                            imageGT_files=imageGT_files_test,
                                            fieldGT_files=fieldGT_files_test,
                                            weightGT_files=maskGT_files_test,
                                            ped=ped,
                                            unsup=unsup,
                                            batch_size=batch_size)

sample_train = next(gen_train)

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

# unet architecture
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]
# enc_nf = [8, 16, 32, 64]
# dec_nf = [64, 32, 16, 8, 8]

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
    

unsup_loss_func = sudistoc.losses.MSE().loss 
sup_loss_func = sudistoc.losses.wMSE().loss 

lr = 1e-3
epsilon = 0.00001

img_weight = 19975
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
    weights += [12]
    if image2_files is not None:
        losses += [sup_loss_func]
        weights += [12]
        
losses += [voxelmorph.losses.Grad('l2', loss_mult=int_downsize).loss]
weights += [1] 
weights = [float(weights[i]/sum(weights)) for i in range(len(weights))]

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, epsilon=epsilon), loss=losses, loss_weights=weights)

# model.summary(line_length=130); print(' ')
KU.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, show_dtype=True, expand_nested=True,) 

print('Model parameters:')
sudistoc.utils.model_params(is_kissing=image2_files is not None, is_unsup=unsup, is_imageGT=imageGT_files is not None, is_fieldGT=fieldGT_files is not None, losses=losses.copy(), weights=weights.copy(), lr=lr, enc_nf=enc_nf, dec_nf=dec_nf)

trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
print('\nNumber of trainable parameters: ',trainableParams)

#%%

initial_epoch = 0
steps_per_epoch = n_train // batch_size
val_steps = n_val // batch_size

model_dir = 'models/' + model_name
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

#%%

best_model = sudistoc.networks.sudistoc_net.load(os.path.join(model_dir, model_name + '.h5'))

x = next(gen_test)
y = best_model.register()(x[0])
z = best_model.predict(x[0])
z = best_model.register2(x[0][0],x[0][1])
zz = [(z[2]-y[2])**2, (z[3]-y[3])**2]

sudistoc.utils.develop(x)
sudistoc.utils.develop(y)
sudistoc.utils.plot_input_GT([x[0],y,z,zz],rowtype=['Input','Output','pred','tg'])
plt.show()

l = sudistoc.losses.MSE().loss(y[2], x[1][1]) 

#%%
dataset = "possum"

image1_files = sorted(glob.glob("datasets/" + dataset + "/*_ap.nii.gz"))
image2_files = sorted(glob.glob("datasets/" + dataset + "/*pa.nii.gz"))
imageGT_files = sorted(glob.glob("datasets/" + dataset + "/*sim.nii.gz"))
fieldGT_files = sorted(glob.glob("datasets/" + dataset + "/*field.nii.gz"))
maskGT_files = None #sorted(glob.glob("datasets/" + dataset + "/*weight.nii.gz"))

image1_files_test = [image1_files[i] for i in ind[n_train+n_val:n_train+n_val+n_test]] 
image2_files_test = [image2_files[i] for i in ind[n_train+n_val:n_train+n_val+n_test]] 
imageGT_files_test = [imageGT_files[i] for i in ind[n_train+n_val:n_train+n_val+n_test]] 
fieldGT_files_test = [fieldGT_files[i] for i in ind[n_train+n_val:n_train+n_val+n_test]]
maskGT_files_test = None #[maskGT_files[i] for i in ind[n_train+n_val:n_train+n_val+n_test]]

gen_test = sudistoc.generators.sudistoc_gen(image1_files=image1_files_test,
                                            image2_files=image2_files_test,
                                            imageGT_files=imageGT_files_test,
                                            fieldGT_files=fieldGT_files_test,
                                            weightGT_files=maskGT_files_test,
                                            ped=ped,
                                            unsup=True,
                                            batch_size=1)

metric_field = np.zeros(n_test)
metric_image = np.zeros(n_test)

for i in range (n_test):
    
    test_in = next(gen_test)
    test_out = best_model.register()(test_in[0])
    
    # ground truth
    image_GT = test_in[1][1][0, ...]
    field_GT = test_in[1][3][0, ...]
    weight = None #test_in[2][1][0, ...]

    # estimated
    image1= test_out[0][0, ...]
    image2 = test_out[1][0, ...]
    field = test_out[2][0, ...]

    metric_field[i] = sudistoc.losses.MSE().loss(field_GT, field, weight) 

    metric_image[i] = (sudistoc.losses.MSE().loss(image_GT, image1, weight) 
                           + sudistoc.losses.MSE().loss(image_GT, image2, weight)) / 2  

    plt.figure(figsize=(20, 5), dpi=80)
    sudistoc.utils.plot_img([np.expand_dims(image_GT,0),np.expand_dims((image1+image2)/2,0),np.expand_dims(field_GT,0),np.expand_dims(field,0)],colnames=['ground truth', 'semi-supervised','ground truth', 'semi-supervised'], cmin=[0, 0, -7, -7], cmax=[0.7, 0.7, 7, 7])
    plt.show()
    