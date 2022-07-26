import numpy as np
import SimpleITK as sitk

def sudistoc_gen(image1_files, 
                 image2_files=None,
                 imageGT_files=None,
                 fieldGT_files=None,
                 weightGT_files=None,
                 ped=None,
                 unsup=True,
                 unsup_isweighted = False,
                 n_downsamplings = 4,
                 batch_size=1):
    
    if weightGT_files is None and unsup_isweighted==True:
        raise ValueError('No weights.')
    if unsup and image2_files is None:
        raise ValueError('Unsupervised setting impossible with single input image.')
    if imageGT_files is None and fieldGT_files is None and not unsup:
        raise ValueError('Activate unsupervised or gives ground truths for supervision (or both).')
            
    while True:

        ind_batch = np.random.choice(range(0, len(image1_files)), size=batch_size, replace=False)
        minmaxFilter = sitk.MinimumMaximumImageFilter()
        intensityFilter = sitk.IntensityWindowingImageFilter() # to have image intensities between 0 and 1.
        intensityFilter.SetWindowMinimum(0)
        intensityFilter.SetOutputMaximum(0)
        intensityFilter.SetOutputMaximum(1)
        
        image1 = [pad_image(sitk.Cast(sitk.ReadImage(image1_files[i]), sitk.sitkFloat32), n_downsamplings) for i in ind_batch]
        if image2_files is not None: image2 = [pad_image(sitk.Cast(sitk.ReadImage(image2_files[i]), sitk.sitkFloat32), n_downsamplings) for i in ind_batch]
        if imageGT_files is not None: imageGT = [pad_image(sitk.Cast(sitk.ReadImage(imageGT_files[i]), sitk.sitkFloat32), n_downsamplings) for i in ind_batch]
           
        for i in range(batch_size):
            minmaxFilter.Execute(image1[i])
            maxi1 = minmaxFilter.GetMaximum()
            maxi2 = 0
            if image2_files is not None:
                minmaxFilter.Execute(image2[i])
                maxi2 = minmaxFilter.GetMaximum()               
            intensityFilter.SetWindowMaximum(max(maxi1,maxi2))
            image1[i] = intensityFilter.Execute(image1[i])
            image1[i] = sitk.GetArrayFromImage(image1[i])[np.newaxis,..., np.newaxis]        
            if image2_files is not None: 
                image2[i] = intensityFilter.Execute(image2[i]) 
                image2[i] = sitk.GetArrayFromImage(image2[i])[np.newaxis,..., np.newaxis]           
            if imageGT_files is not None:
                imageGT[i] = intensityFilter.Execute(imageGT[i]) 
                imageGT[i] = sitk.GetArrayFromImage(imageGT[i])[np.newaxis,..., np.newaxis]
        
        image1 = np.concatenate(image1, axis=0)   
        if image2_files is not None:
            image2 = np.concatenate(image2, axis=0)
        if imageGT_files is not None:
            imageGT = np.concatenate(imageGT, axis=0)              
        if fieldGT_files is not None: 
            fieldGT = [sitk.GetArrayFromImage(pad_image(sitk.Cast(sitk.ReadImage(fieldGT_files[i]), sitk.sitkFloat32), n_downsamplings))[np.newaxis,..., np.newaxis] for i in ind_batch]        
            fieldGT = np.concatenate(fieldGT, axis=0)
        if weightGT_files is not None:  # readout = 0.052071 
            weightGT = [sitk.GetArrayFromImage(pad_image(sitk.Cast(sitk.ReadImage(weightGT_files[i]), sitk.sitkFloat32), n_downsamplings))[np.newaxis,..., np.newaxis] for i in ind_batch]        
            weightGT = np.concatenate(weightGT, axis=0)
             
        inshape = np.shape(image1)[1:-1]
        ndims = len(inshape)
        is_weight = weightGT_files is not None
          
        if unsup:
            image0 = np.zeros_like(image1)
        field0 = np.zeros((batch_size, *inshape, ndims))
        
        inputs = [image1]
        if image2_files is not None:
            inputs += [image2]
            
        groundTruths = []
        
        if unsup:
            if is_weight and unsup_isweighted:
                groundTruths += [np.concatenate([image0, weightGT], axis=-1)]
            else:
                groundTruths += [image0]          
       
        if imageGT_files is not None:
            if is_weight:
                groundTruths += [np.concatenate([imageGT, weightGT], axis=-1)]
            else:
                groundTruths += [imageGT]
            if image2_files is not None:
                if is_weight:
                    groundTruths += [np.concatenate([imageGT, weightGT], axis=-1)]
                else:
                    groundTruths += [imageGT]
        
        if fieldGT_files is not None:
            if is_weight:
                groundTruths += [np.concatenate([-fieldGT, weightGT], axis=-1)]
            else:
                groundTruths += [-fieldGT] 
            if image2_files is not None:
                if is_weight:
                    groundTruths += [np.concatenate([fieldGT, weightGT], axis=-1)]
                else:
                    groundTruths += [fieldGT] 
                
        groundTruths += [field0]   

        yield (inputs, groundTruths) 


      
def pad_image(img, n_downsamplings):
    """
    Pad an image such that image size along each dimension becomes of form k*2^n.
    Adapted for networks with less or equal to n down/upsamplings.
    """
    
#    outSize = np.power(2, np.ceil(np.log(img.GetSize())/np.log(2)))
    outSize = np.ceil(np.array(img.GetSize()) / 2**n_downsamplings) * 2**n_downsamplings
    lowerPad = np.round((outSize - img.GetSize()) / 2)
    upperPad = outSize - img.GetSize() - lowerPad

    padder = sitk.ConstantPadImageFilter()
    padder.SetConstant(0)
    padder.SetPadLowerBound(lowerPad.astype(int).tolist())
    padder.SetPadUpperBound(upperPad.astype(int).tolist())
    
    paddedImg = padder.Execute(img)
    
    return paddedImg
