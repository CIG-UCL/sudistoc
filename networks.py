"""
tensorflow/keras networks for voxelmorph

If you use this code, please cite one of the voxelmorph papers:
https://github.com/voxelmorph/voxelmorph/blob/master/citations.bib

License: GPLv3
"""

# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI

import voxelmorph as vxm

# local imports
import neurite as ne
from . import layers 
from . import modelio


class sudistoc_net(modelio.LoadableModel):
    """
    sudistoc network for suceptibility distortion correction.
    """

    @vxm.tf.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 input_model=None,              
                 kissing=True,
                 constraint='oppsym',  # 'oppsym' (opposite symmetry) or 'diffeo' (inverse symmetry)
                 ped=None,   # axis corresponding to the phase encoding direction. If None, unconstrained registration.
                 jacob_mod=True,   # jacobian intensity modulation               
                 transfo_sup=False,  # supervised with ground truth transformations
                 image_sup=False,  # supervised with ground truth image
                 unsup=True,  # unsupervised similarity between undistorted images
                 name='sudistoc_net'):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            unet_half_res: Skip the last unet decoder upsampling. Requires that int_downsize=2. Default is False.
            input_model: Model to replace default input layer before concatenation. Default is None.
            name: Model name - also used as layer name prefix. Default is 'vxm_dense'.
        """
        
        # ensure compatible settings
        if unsup and not kissing:
            raise ValueError('Unsupervised setting impossible with single image (no kissing)')
        if not transfo_sup and not image_sup and not unsup:
            raise ValueError('At least one of transfo_sup, image_sup or unsup has to be True')


        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_input' % name)
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_input' % name)
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # build core unet model and grab inputs
        unet_model = vxm.networks.Unet(input_model=input_model,
                                       nb_features=nb_unet_features,
                                       nb_levels=nb_unet_levels,
                                       feat_mult=unet_feat_mult,
                                       nb_conv_per_level=nb_unet_conv_per_level,
                                       half_res=unet_half_res)

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        if ped is None:
            flow_mean = Conv(ndims, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_flow' % name)(unet_model.output)
        else:
            flow_mean = Conv(1, kernel_size=3, padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5), name='%s_flow' % name)(unet_model.output)
            flow_zeros = tf.keras.backend.zeros_like(flow_mean)
            
            flow_list = []  
            for i in range(0, ndims):
                if i == ped:
                    flow_list.append(flow_mean)
                else:
                    flow_list.append(flow_zeros)
             
            flow_mean = KL.concatenate(flow_list, name='%s_concat_flow' % name)
            
        # optionally include probabilities
        if use_probs:
            # initialize the velocity variance very low, to start stable
            flow_logsigma = Conv(ndims, kernel_size=3, padding='same',
                            kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-10),
                            bias_initializer=KI.Constant(value=-10),
                            name='%s_log_sigma' % name)(unet_model.output)
            flow_params = KL.concatenate([flow_mean, flow_logsigma], name='%s_prob_concat' % name)
            flow = ne.layers.SampleNormalLogVar(name='%s_z_sample' % name)([flow_mean, flow_logsigma])
        else:
            flow_params = flow_mean
            flow = flow_mean

        if not unet_half_res:
            # optionally resize for integration
            if int_steps > 0 and int_downsize > 1:
                flow = vxm.layers.RescaleTransform(1 / int_downsize, name='%s_flow_resize' % name)(flow)

        preint_flow = flow

        pos_flow = flow
        
        if kissing and constraint == 'diffeo':
            neg_flow = ne.layers.Negate(name='%s_neg_flow' % name)(flow)            

        # integrate to produce diffeomorphic warp (i.e. treat flow as a stationary velocity field)
        if int_steps > 0:
            pos_flow = vxm.layers.VecInt(method='ss', name='%s_flow_int' % name, int_steps=int_steps)(pos_flow)
            if kissing and constraint == 'diffeo':
                neg_flow = vxm.layers.VecInt(method='ss', name='%s_neg_flow_int' % name, int_steps=int_steps)(neg_flow)
                
            # resize to final resolution
            if int_downsize > 1:
                pos_flow = vxm.layers.RescaleTransform(int_downsize, name='%s_diffflow' % name)(pos_flow)
                if kissing and constraint == 'diffeo':
                    neg_flow = vxm.layers.RescaleTransform(int_downsize, name='%s_neg_diffflow' % name)(neg_flow)
                    
        if constraint == 'oppsym':
            neg_flow = ne.layers.Negate(name='%s_neg_transfo' % name)(pos_flow)               


        # warp image with flow field
        y_source = vxm.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_transformer' % name)([source, pos_flow])
        if kissing:
            y_target = vxm.layers.SpatialTransformer(interp_method='linear', indexing='ij', name='%s_neg_transformer' % name)([target, neg_flow])
        if jacob_mod:
            y_source = layers.JacobianMultiplyIntensities(indexing='ij', name='%s_det_Jac_multiply_source' % name)([y_source, pos_flow])
            if kissing:
                y_target = layers.JacobianMultiplyIntensities(indexing='ij', name='%s_det_Jac_multiply_target' % name)([y_target, neg_flow])
        
        if ped is not None:
            pos_flow_dir = layers.Slice(index=ped, name='%s_slice_pos_transfo' % name)(pos_flow)
            neg_flow_dir = layers.Slice(index=ped, name='%s_slice_neg_transfo' % name)(neg_flow)
        else:
            pos_flow_dir = pos_flow
            neg_flow_dir = neg_flow
            
        outputs=[]
        
        if unsup:
            y_diff = KL.Subtract(name='unsup')([y_target, y_source])
            outputs += [y_diff]
         
        if image_sup:
            dummy_layer = KL.Lambda(lambda x: x, name='sup_image')
            y_source = dummy_layer(y_source)
            outputs += [y_source]
            if kissing: 
                dummy_layer = KL.Lambda(lambda x: x, name='sup_image_neg')
                y_target = dummy_layer(y_target)
                outputs += [y_target]
         
        if transfo_sup:
            dummy_layer = KL.Lambda(lambda x: x, name='sup_field')
            pos_flow_dir = dummy_layer(pos_flow_dir)
            outputs += [pos_flow_dir]
            if kissing:
                dummy_layer = KL.Lambda(lambda x: x, name='sup_field_neg')
                neg_flow_dir = dummy_layer(neg_flow_dir)
                outputs += [neg_flow_dir]
         
        if use_probs:
            # compute loss on flow probabilities
            dummy_layer = KL.Lambda(lambda x: x, name='smooth')
            flow_params = dummy_layer(flow_params)
            outputs += [flow_params]
        else:
            # compute smoothness loss on pre-integrated warp
            dummy_layer = KL.Lambda(lambda x: x, name='smooth')
            preint_flow = dummy_layer(preint_flow)
            outputs += [preint_flow]
          
        super().__init__(name=name, inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = vxm.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.kissing = kissing
        self.references.unsup = unsup
        self.references.y_diff = y_diff
        self.references.y_source = y_source
        self.references.y_target = y_target if kissing else None
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow if kissing else None
        self.references.pos_flow_dir = pos_flow_dir
        self.references.neg_flow_dir = neg_flow_dir
        
    # def get_registration_model(self):
    #     """
    #     Returns a reconfigured model to predict only the final transform.
    #     """
    #     return tf.keras.Model(self.inputs[:2], self.references.y_source)
    
    def register(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        if self.references.kissing:
            return tf.keras.Model(self.inputs, [self.references.y_source, self.references.y_target, self.references.pos_flow_dir, self.references.neg_flow_dir])
        else:
            return tf.keras.Model(self.inputs, [self.references.y_source, self.references.pos_flow])

    # def register(self, src, trg):
    #     """
    #     Predicts the transform from src to trg tensors.
    #     """
    #     return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm.layers.SpatialTransformer(interp_method=interp_method)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])
