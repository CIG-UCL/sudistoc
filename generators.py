import numpy as np
import nibabel as nib
    
def sudistoc_gen(image1_files, 
                 image2_files=None,
                 imageGT_files=None,
                 fieldGT_files=None,
                 weightGT_files=None,
                 ped=None,
                 unsup=True,
                 batch_size=1):
  
    if unsup and image2_files is None:
        raise ValueError('Unsupervised setting impossible with single input image')
    if imageGT_files is None and fieldGT_files is None and not unsup:
        raise ValueError('Activate unsupervised or gives ground truths for supervision (or both)')
            
    while True:

        ind_batch = np.random.choice(range(0, len(image1_files)), size=batch_size, replace=False)

        image1 = [nib.load(image1_files[i]).get_fdata().squeeze()[np.newaxis,..., np.newaxis] for i in ind_batch]
        image1 = np.concatenate(image1, axis=0)
        image1 = image1.astype(np.float32)
        
        if image2_files is not None:
            image2 = [nib.load(image2_files[i]).get_fdata().squeeze()[np.newaxis,..., np.newaxis] for i in ind_batch]
            image2 = np.concatenate(image2, axis=0)
            image2 = image2.astype(np.float32)
        
        inshape = np.shape(image1)[1:-1]
        ndims = len(inshape)
        is_weight = weightGT_files is not None
        
        if is_weight:
            weightGT = [nib.load(weightGT_files[i]).get_fdata().squeeze()[np.newaxis,..., np.newaxis] for i in ind_batch]
            weightGT = np.concatenate(weightGT, axis=0)
            weightGT = weightGT.astype(np.float32)
            weights = ()
                    
        if imageGT_files is not None:
            imageGT = [nib.load(imageGT_files[i]).get_fdata().squeeze()[np.newaxis,..., np.newaxis] for i in ind_batch]
            imageGT = np.concatenate(imageGT, axis=0)
            imageGT = imageGT.astype(np.float32)
        
        if fieldGT_files is not None:
    #        readout = 0.052071 
            fieldGT = [nib.load(fieldGT_files[i]).get_fdata().squeeze()[np.newaxis,..., np.newaxis] for i in ind_batch]
            fieldGT = np.concatenate(fieldGT, axis=0)
            fieldGT = fieldGT.astype(np.float32)
                
        if unsup:
            image0 = np.zeros_like(image1)
        field0 = np.zeros((batch_size, *inshape, ndims))
        
        inputs = [image1, image2]
        
        groundTruths = []
        
        if unsup:         
            groundTruths += [image0]
            if is_weight:
                weights += (np.ones_like(weightGT),)#  (None,)
       
        if imageGT_files is not None:
            groundTruths += [imageGT]
            if is_weight:
                weights += (weightGT,)
            if image2_files is not None:
                groundTruths += [imageGT]
                if is_weight:
                    weights += (weightGT,)
        
        if fieldGT_files is not None:
            groundTruths +=  [fieldGT]    
            if is_weight:
                weights += (weightGT,)
            if image2_files is not None:
                groundTruths += [-fieldGT]
                if is_weight:
                    weights += (weightGT,)
                
        groundTruths += [field0]   
        if is_weight:
            weights += (np.ones_like(weightGT),)#  (None,)
        
        if is_weight:
            yield (inputs, groundTruths, weights)  
        else:
            yield (inputs, groundTruths) 