import torch.nn as nn                                                           # This is the psuedocode for the RfR implimentation of Dr. Guo's RRN MRI Registration project. 
import torch                                                                    # {} : refers to another file in the repo/directory
import torch.nn.functional as func                                              # [] : refers to a line in the current file, unless prefixed with {} or comma-seperated
import gpu_utils                                                                # [,]: refers to a tuple, list, array, matrix, tensor, etc.
                                                                                # Variables are refered to by their handle name
import rrn_utils
import RfR

class RRNFlow(nn.Module):
    def __init__(self,
                 leaky_relu_alpha=0.05,
                 drop_out_rate=0,
                 num_context_up_channels = 32,
                 num_levels=5, 
                 normalize_before_cost_volume=True,
                 channel_multiplier=1.,
                 use_cost_volume=True,
                 use_feature_warp=True,
                 accumulate_flow=True,
                 shared_flow_decoder=False,
                 action_channels=None):
        super(RRNFlow, self).__init__()

        self._leaky_relu_alpha = leaky_relu_alpha
        self._drop_out_rate = drop_out_rate
        self._num_context_up_channels = num_context_up_channels
        self._num_levels = num_levels
        self._normalize_before_cost_volume = normalize_before_cost_volume
        self._channel_multiplier = channel_multiplier
        self._use_cost_volume = use_cost_volume
        self._use_feature_warp = use_feature_warp
        self._accumulate_flow = accumulate_flow
        self._shared_flow_decoder = shared_flow_decoder
        self._action_channels=action_channels

        #self._refine_model = self._build_refinement_model()
        #self._flow_layers = self._build_flow_layers()
        if not self._use_cost_volume:                                                                               # Currently false because use_cost_volume = True, but
            self._cost_volume_surrogate_convs = self._build_cost_volume_surrogate_convs()                           # the function called in this line is defined on [173]
        if num_context_up_channels:
            self._context_up_layers = self._build_upsample_layers(
                num_channels=int(num_context_up_channels * channel_multiplier))
        if self._shared_flow_decoder:
            # pylint:disable=invalid-name
            self._1x1_shared_decoder = self._build_1x1_shared_decoder()

    def forward(self, feature_pyramid1, feature_pyramid2, actions=None):
        """Run the model."""
        context = None
        flow = None
        flow_up = None
        context_up = None
        flows = []

        # make sure that actions are provided iff network is configured for action use                              # This block is currently irrelevant because actions and 
        #if actions is not None:                                                                                    # action_channels are both None. No need to check or assert
        #    assert self._action_channels == actions.shape[1]                                                       # anything, and we may want to consolidate the two variables.
        #else:
        #    assert self._action_channels is None

        # Go top down through the levels to the second to last one to estimate flow.
        for level, (features1, features2) in reversed(
                list(enumerate(zip(feature_pyramid1, feature_pyramid2)))[1:]):

            # init flows with zeros for coarsest level if needed
            if self._shared_flow_decoder and flow_up is None:
                batch_size, height, width, _ = features1.shape.as_list()
                flow_up = torch.zeros([batch_size, height, width, 2]).to(gpu_utils.device)
                if self._num_context_up_channels:                                                                   # Currently always true
                    num_channels = int(self._num_context_up_channels *                                              # Channel multiplier is 1 so this does nothing
                                       self._channel_multiplier)                                                    
                    context_up = torch.zeros([batch_size, height, width, num_channels]).to(gpu_utils.device)

            # Warp features2 with upsampled flow from higher level.
            if flow_up is None or not self._use_feature_warp:
                warped2 = features2
            else:
                warp_up = rrn_utils.flow_to_warp(flow_up)
                warped2 = rrn_utils.resample(features2, warp_up)

            # Compute cost volume by comparing features1 and warped features2.
            features1_normalized, warped2_normalized = rrn_utils.normalize_features(
                [features1, warped2],
                normalize=self._normalize_before_cost_volume,
                center=self._normalize_before_cost_volume,
                moments_across_channels=True,
                moments_across_images=True)

            if self._use_cost_volume:
                cost_volume = rrn_utils.compute_cost_volume(features1_normalized, warped2_normalized, max_displacement=2)
            else:
                concat_features = torch.cat([features1_normalized, warped2_normalized], dim=1)
                cost_volume = self._cost_volume_surrogate_convs[level](concat_features)

            cost_volume = func.leaky_relu(cost_volume, negative_slope=self._leaky_relu_alpha)

            if self._shared_flow_decoder:
                # This will ensure to work for arbitrary feature sizes per level.
                conv_1x1 = self._1x1_shared_decoder[level]
                features1 = conv_1x1(features1)

            # Compute context and flow from previous flow, cost volume, and features1.                                                  
            if flow_up is None:
                x_in = torch.cat([cost_volume, features1], dim=1)
            else:
                if context_up is None:
                    x_in = torch.cat([flow_up, cost_volume, features1], dim=1)                                                          # The input to the DVF estimator, Fig 2(b)
                else:
                    x_in = torch.cat([context_up, flow_up, cost_volume, features1], dim=1)                                              # x_in is the input set to the DVF estimator

            if self._action_channels is not None and self._action_channels > 0:                                                         # Currently irrellevant because action_channels
                # convert every entry in actions to a channel filled with this value and attach it to flow input                        # is None
                B, _, H, W = features1.shape
                # additionally append xy position augmentation
                action_tensor = actions[:, :, None, None].repeat(1, 1, H, W)
                gy, gx = torch.meshgrid([torch.arange(H).float(), torch.arange(W).float()])
                gx = gx.repeat(B, 1, 1, 1).to(gpu_utils.device)
                gy = gy.repeat(B, 1, 1, 1).to(gpu_utils.device)
                x_in = torch.cat([x_in, action_tensor, gx, gy], dim=1)
                
            
            # Use dense-net connections.
            x_out = None
            if self._shared_flow_decoder:
                # reuse the same flow decoder on all levels
                flow_layers = self._flow_layers
            else:
                flow_layers = self._flow_layers[level]
            for layer in flow_layers[:-1]:
                x_out = layer(x_in)
                x_in = torch.cat([x_in, x_out], dim=1)
            context = x_out

            flow = flow_layers[-1](context)                                                                               
                      
            # dropout full layer
            if self.training and self._drop_out_rate:
                maybe_dropout = (torch.rand([]) > self._drop_out_rate).type(torch.get_default_dtype())
                # note that operation must not be inplace, otherwise autograd will fail pathetically
                context = context * maybe_dropout
                flow = flow * maybe_dropout

            if flow_up is not None and self._accumulate_flow:
                flow += flow_up

            # Upsample flow for the next lower level.
            flow_up = rrn_utils.upsample(flow, is_flow=True)
            if self._num_context_up_channels:
                context_up = self._context_up_layers[level](context)

            # Append results to list.
            flows.insert(0, flow)

        # Refine flow at level 1.
        # refinement = self._refine_model(torch.cat([context, flow], dim=1))
        refinement = torch.cat([context, flow], dim=1)
        
        last_in_channels = 32+3                                                                     # The number of inputs to the final DVF is len(context) + len(flow) = 32+3
        DVF_est = RfR.RfR_model('RunfastReg', last_in_channels, 3)       
        cuda = False
        if cuda == True:
            DVF_est.cuda()
        else:
            DVF_est
        with torch.no_grad():
            if cuda == True:
                refinement = DVF_est(refinement).cuda()
            else:
                refinement = DVF_est(refinement)
        
        # for layer in self._refine_model:
            # refinement = layer(refinement)

        # dropout refinement
        if self.training and self._drop_out_rate:
            maybe_dropout = (torch.rand([]) > self._drop_out_rate).type(torch.get_default_dtype())
            # note that operation must not be inplace, otherwise autograd will fail pathetically
            refinement = refinement * maybe_dropout

        refined_flow = flow + refinement
        flows[0] = refined_flow

        flows.insert(0, rrn_utils.upsample(flows[0], is_flow=True))
        flows.insert(0, rrn_utils.upsample(flows[0], is_flow=True))

        return flows

    def _build_cost_volume_surrogate_convs(self):                                                   # This is used if use_cost_volume is not true, which is currently is. {rrn_net.py}
        layers = nn.ModuleList()
        for _ in range(self._num_levels):
            layers.append(nn.Sequential(
                nn.ZeroPad3d((2,1,2,1,2,1)), # should correspond to "SAME" in keras
                nn.Conv3d(
                    in_channels=int(64 * self._channel_multiplier),                                 # Currently 64 because channel_multiplier = 1
                    out_channels=int(64 * self._channel_multiplier),                                # Currently 64 because channel_multiplier = 1
                    kernel_size=(4, 4)))
            )
        return layers

    def _build_upsample_layers(self, num_channels):
        """Build layers for upsampling via deconvolution."""
        layers = []
        for unused_level in range(self._num_levels):
            layers.append(
                nn.ConvTranspose3d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=(4, 4, 4),
                    stride=2,
                    padding=1))
        return nn.ModuleList(layers)

    def _build_flow_layers(self):                                                                   # The following is for Figure 2d. We are using CNN layers to estimate initial and intermdiate DVF;
        """Build layers for flow estimation."""                                                     # The original implimentation used an RNN but we have replaced it with 
        # Empty list of layers level 0 because flow is only estimated at levels > 0.                # a modified UNet called RunfastReg (RfR). 
        # result = nn.ModuleList()
        out = []
        for i in range(0, self._num_levels):                                                        # Loop moves through each level of the RRN (NOT THE DVF)
            #layers = nn.ModuleList()                                                               # Context currently has 32 layers
            last_in_channels = (64+32)#if not self._use_cost_volume else (125+input_feat_shape[i])  # Number of channels into final RfR Layer; _use_cost_volume=True (line 16)
            #if self._action_channels is not None and self._action_channels > 0:                    # This condition is never true because _action_channels is forced None
            #    last_in_channels += self._action_channels + 2                                      # (2 for xy augmentation)?
            if i != self._num_levels-1:                                                             # if building intermediate - as opposed to initial - DVF (num_layers = 5),
                last_in_channels += 3 + self._num_context_up_channels                               # then DVF input is 96 (LCV) + 3 (flow from previous layer) + 32(upsampled feature 1?)
            cuda = False    
            DVF_est = RfR.RfR_model('RunfastReg', last_in_channels, _num_context_up_channels)       
            if cuda == True:
                DVF_est.cuda()
            else:
                DVF_est
            #
            with torch.no_grad():
                if cuda == True:
                    out = DVF_est(x_in).cuda()
                else:
                    out = DVF_est(x_in)
        return out

    def _build_refinement_model(self, refinement):                                                   # This is the code used to create the final DVF estimator in accordance 
        #Build model for flow refinement using dilated convolutions.                                 # with Fig.2(e).
        out = []
        last_in_channels = 32+3                                                                     # The number of inputs to the final DVF is len(context) + len(flow) = 32+3
        DVF_est = RfR.RfR_model('RunfastReg', last_in_channels, 3)       
        cuda = False
        if cuda == True:
            DVF_est.cuda()
        else:
            DVF_est
        with torch.no_grad():
            if cuda == True:
                out = DVF_est(refinement).cuda()
            else:
                out = DVF_est(refinement)
        return out

    def _build_1x1_shared_decoder(self):
        """Build layers for flow estimation."""
        # Empty list of layers level 0 because flow is only estimated at levels > 0.
        result = nn.ModuleList([nn.ModuleList()])
        for _ in range(1, self._num_levels):
            result.append(
                nn.Conv3d(
                    in_channels= 32,
                    out_channels=32,
                    kernel_size=(1, 1),
                    stride=1))
        return result

class RRNFeaturePyramid(nn.Module):
  """Model for computing a feature pyramid from an image."""

  def __init__(self,
               leaky_relu_alpha=0.1,
               filters=None,
               level1_num_layers=3,
               level1_num_filters=32,
               level1_num_1x1=0,
               original_layer_sizes=True,
               num_levels=5,
               channel_multiplier=1.,
               pyramid_resolution='half',
               num_channels=3):
    super(RRNFeaturePyramid, self).__init__()

    self._channel_multiplier = channel_multiplier
    if num_levels > 6:
      raise NotImplementedError('Max number of pyramid levels is 6')
    if filters is None:
      if original_layer_sizes:
        # Orig - last layer
        filters = ((3, 16), (3, 32), (3, 64), (3, 96), (3, 128),
                   (3, 196))[:num_levels]
      else:
        filters = ((level1_num_layers, level1_num_filters), (3, 32), (3, 32),
                   (3, 32), (3, 32), (3, 32))[:num_levels]
    assert filters
    assert all(len(t) == 2 for t in filters)
    assert all(t[0] > 0 for t in filters)

    self._leaky_relu_alpha = leaky_relu_alpha
    self._convs = nn.ModuleList()
    self._level1_num_1x1 = level1_num_1x1

    c = num_channels

    for level, (num_layers, num_filters) in enumerate(filters):
      group = nn.ModuleList()
      for i in range(num_layers):
        stride = 1
        if i == 0 or (i == 1 and level == 0 and pyramid_resolution == 'quarter'):
          stride = 2
        conv = nn.Conv3d(
            in_channels=c,
            out_channels=int(num_filters * self._channel_multiplier),
            kernel_size=3 if level > 0 or i < num_layers - level1_num_1x1 else 1,
            stride=stride,
            padding=0)
        group.append(conv)
        c = int(num_filters * self._channel_multiplier)
      self._convs.append(group)

  def forward(self, x, split_features_by_sample=False):
    x = x * 2. - 1.  # Rescale input from [0,1] to [-1, 1]
    features = []
    for level, conv_tuple in enumerate(self._convs):
      for i, conv in enumerate(conv_tuple):
        if level > 0 or i < len(conv_tuple) - self._level1_num_1x1:
          x = func.pad(
              x,
              pad=[1, 1, 1, 1,1,1],
              mode='constant',
              value=0)
        x = conv(x)
        x = func.leaky_relu(x, negative_slope=self._leaky_relu_alpha)
      features.append(x)

    if split_features_by_sample:

      # Split the list of features per level (for all samples) into a nested
      # list that can be indexed by [sample][level].

      n = len(features[0])
      features = [[f[i:i + 1] for f in features] for i in range(n)]  # pylint: disable=g-complex-comprehension

    return features

""" This is the original init/interm DVF portion which I have commented for my own uses - the comments might be useful for future implimentations of the RRN with or without RfR
    def _build_flow_layers(self):
        # Build layers for flow estimation.
        # Empty list of layers level 0 because flow is only estimated at levels > 0.
        result = nn.ModuleList()

        block_layers = [128, 128, 96, 64, 32]
        input_feat_shape = [196,128,96,64,32,16][::-1][:4]
        for i in range(0, self._num_levels):
            layers = nn.ModuleList()
            last_in_channels = (64+32) if not self._use_cost_volume else (125+input_feat_shape[i])
            if self._action_channels is not None and self._action_channels > 0:
                last_in_channels += self._action_channels + 2 # 2 for xy augmentation
            if i != self._num_levels-1:
                last_in_channels += 3 + self._num_context_up_channels
            for c in block_layers:                                                                  # This loop creates the final/interm DVF by appending 3D CNN layers in
                layers.append(                                                                      # sequence (using the length and elements of block_layers to determine the 
                    nn.Sequential(                                                                  # depth and channel outputs respectively) according to Fig.2(d).
                        nn.Conv3d(
                            in_channels=last_in_channels,                                           # last_in_channels = 64+32
                            out_channels=int(c * self._channel_multiplier),                         # c cycles block layers;  _channel_multiplier = 1.
                            kernel_size=(3, 3, 3),
                            padding=1),
                        nn.LeakyReLU(
                            negative_slope=self._leaky_relu_alpha)
                    ))
                last_in_channels += int(c * self._channel_multiplier)
            layers.append(                                                                          # The final layer of the DVF is appended to the DVF sequence here.
                nn.Conv3d(
                    in_channels=block_layers[-1],
                    out_channels=3,
                    kernel_size=(3, 3, 3),
                    padding=1))
            if self._shared_flow_decoder:
                return layers
            result.append(layers)
        return result
[end of snippet]"""

""" This is the original final DVF portion which I have commented for my own uses - the comments might be useful for future implimentations of the RRN with or without RfR
    def _build_refinement_model(self):                                                              # This is the code used to create the final DVF estimator in accordance 
        #Build model for flow refinement using dilated convolutions.                                 # with Fig.2(e).
        layers = []
        last_in_channels = 32+3                                                                     # The number of inputs to the final DVF is len(context) + len(flow) = 32+3
        for c, d in [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]:                     # List of tuples: (each layer's output channel size, dilation)
            layers.append(
                nn.Conv3d(
                    in_channels=last_in_channels,
                    out_channels=int(c * self._channel_multiplier),
                    kernel_size=(3, 3,3),
                    stride=1,
                    padding=d,
                    dilation=d))
            layers.append(
                nn.LeakyReLU(negative_slope=self._leaky_relu_alpha))
            last_in_channels = int(c * self._channel_multiplier)
        layers.append(
            nn.Conv3d(
                in_channels=last_in_channels,
                out_channels=3,
                kernel_size=(3, 3,3),
                stride=1,
                padding=1))
        return nn.ModuleList(layers)
[end of snippet]"""