'''
Deep learning model

version 1
4/13/2020
'''

from madmax.api import *

mxtr = mxtr.reload()

MONITOR_PATIENCE = 15

ACTIVATIONS = dict(
    swish=mxtr.Swish,
    tanh=tr.nn.Tanh,
    relu=tr.nn.ReLU,
    leaky_relu=partial( tr.nn.LeakyReLU, negative_slope=0.1 ),
    leaky_tanh=mxtr.LeakyTanh,
    linear=tr.nn.Identity,
    identity=tr.nn.Identity,
    none=tr.nn.Identity,
)


class OneLayer( mxtr.Module ):
    '''
    one layer of the resnet
    
    Parameters
    ----------
    layer:
        the name or number of this layer
    Pin
        num features
    Pout
        if None - same as Pin
    L:
        length of the convolution kernel. If None or 0 - plain linear network
    activation:
        From ACTIVATIONS
    batch_norm: bool
        if True, 1d batch normalization
    '''
    
    def __init__( self, name, L, Pin, Pout=None, dilation=1, activation='relu',
                  batch_norm=False, monitor=False, grad_norm=None ):
        super().__init__()
        self.name = name
        self.L = L
        self.Pin = Pin
        self.Pout = Pout if Pout is not None else Pin
        self.dilation = dilation
        if L and L > 1:
            self.convolution = mxtr.CausalConv1D( L=self.L, Pin=self.Pin, Pout=self.Pout, bias=True,
                                                  dilation=self.dilation )
        
        else:
            # use a plain linear network if L is None or )
            self.convolution = mxtr.Linear( in_features=self.Pin, out_features=self.Pout, bias=True )
        # Why do you need this affine shit ? It standardizes the data and then rescales and shifts it. Why ?
        self.batch_norm = (mxtr.BatchNormPanel( num_features=self.Pout, affine=True )
                           if batch_norm else tr.nn.Identity())
        self.activation = ACTIVATIONS[activation]() if activation else tr.nn.Identity()
        self.monitor = mxtr.FlowMonitor( name=self.name, patience=MONITOR_PATIENCE ) if monitor else tr.nn.Identity()
        self.grad_norm = grad_norm  # TODO
        self.initialize()
    
    def forward( self, input ):
        '''note - batch norm is before the activation - delegate to activation_regularization'''
        reg, output = self.activation_regularization( input )
        return output
    
    def activation_regularization( self, input, kapow=None ):
        '''apply the activation regularization before the non-linearity'''
        output = self.convolution( input )
        reg = mxtr.regularizations.kapow( output, penalty=kapow, eps=1e-6 ) if kapow else 0
        output = self.batch_norm( output )
        output = self.activation( output )
        output = self.monitor( output )
        return reg, output
    
    def initialize( self, method='normal' ):
        '''
        Parameters
        ----------
        method:
            whatever is used by the convolution layer
        '''
        self.convolution.initialize( method )
        self.batch_norm.initialize()
        self.monitor.initialize()
        return self

    def weight_regularization( self, ridge=None, lasso=None, smooth_lasso=None, logdet=None ):
        if mx.issafeinstance( self.convolution, mxtr.CausalConv1D ):
            reg = self.convolution.weight_regularization( ridge=ridge, lasso=lasso, smooth_lasso=smooth_lasso,
                                                          logdet=logdet )
        else:
            reg = self.convolution.weight_regularization( ridge=ridge, lasso=lasso, smooth_lasso=smooth_lasso, )
        reg += self.batch_norm.weight_regularization( ridge=ridge )
        return reg


class FactorResnet( mxtr.Module ):
    '''the resnet for factor learning and prediction
    
    hydra:
        if True - preshapes all the inputs with hydra ...
    loss_hps:
        hps to control the loss functions
    
    
    '''
    
    def __init__( self, num_features, num_responses=1, activation='leaky_relu', hydra=True,
                  L=50, dilation=2, resnet_layers=3, resnet_channels=10, resnet_gating=True,
                  batch_norm=False, monitor=False, loss_hps=None ):
        super().__init__()
        self.num_features = num_features
        self.num_responses = num_responses
        self.L = L
        self.dilation = dilation
        self.activation = activation
        self.resnet_layers = resnet_layers
        self.resnet_channels = resnet_channels
        self.resnet_gating = resnet_gating
        self.batch_norm = batch_norm
        self.monitor = monitor
        self.loss_hps = loss_hps
        # --- build the network ---
        if hydra:
            # preshape every input feature with a OnlineHydra network
            self.hydra = mxtr.Hydra( num_features=self.num_features, prescale=True, prebias=True, postscale=True )
        else:
            self.hydra = None
        self.first_layer = OneLayer( name='first', L=None, Pin=self.num_features, Pout=self.resnet_channels,
                                     activation=activation, batch_norm=self.batch_norm, monitor=self.monitor )
        if self.resnet_layers:
            _layers = []
            for l in range( self.resnet_layers ):
                ol = OneLayer( name=f'resnet_{l}', L=self.L, Pin=self.resnet_channels, Pout=self.resnet_channels,
                               dilation=self.dilation ** l, activation=self.activation,
                               batch_norm=self.batch_norm, monitor=self.monitor )
                _layers.append( ol )
            self.resnet = mxtr.ResNet( layers=_layers, num_channels=self.resnet_channels, gating=self.resnet_gating )
            self._required_priming = self.L * (self.dilation ** self.resnet_layers)
        else:
            self.resnet = None
            self._required_priming = 0
        # test whether the provided priming is enough
        if self.loss_hps.priming < self._required_priming - 100:
            self.warn( f'Provided priming {self.loss_hps.priming} < required priming {self._required_priming}' )
        elif self.loss_hps.priming > self._required_priming + 100:
            self.warn( f'Provided priming {self.loss_hps.priming} > required priming {self._required_priming}' )
        self.final_layer = mxtr.Sequential(
            mxtr.Linear( self.resnet_channels, self.num_responses ),
            mxtr.FlowMonitor( name='final', patience=MONITOR_PATIENCE ) if self.monitor else tr.nn.Identity(),
        )
        # TODO --- add a timing layer
        # TODO --- add a sid embedding layer and sid-attention weights
        # TODO --- add a time attention weighting
        # TODO --- add quantiles to the loss function
        # TODO --- multifactor learning with logdet penatly
        # some loss functions
        self._mse = mxtr.Error( metric='mse', kappa=None )
        self._util = mxtr.Error( metric='util', kappa=1 )
        self._psr = mxtr.Error( metric='pos-sr', kappa=1e-6 )
        self.initialize()
    
    def forward( self, X ):
        '''forward  - delegate to activation regularization to avoid duplicate code paths'''
        return self.activation_regularization( X )[1]
    
    def activation_regularization( self, X, kapow=None, tv=None, curvature=None ):
        '''
        Activation regularization on the resnet and intermediate layers.
        '''
        reg = 0
        if self.hydra is not None:
            _reg, X = self.hydra.activation_regularization( X, kapow=kapow )
            reg += _reg
        _reg, X = self.first_layer.activation_regularization( X, kapow=kapow )
        reg += _reg
        if self.resnet is not None:
            _reg, X = self.resnet.activation_regularization( X, layer_kwargs=dict( kapow=kapow ) )
            reg += _reg
        _reg, X = self.final_layer.activation_regularization( X, tv=tv, curvature=curvature )
        reg += _reg
        return reg, X
    
    def initialize( self, method='exp_wavelets' ):
        if self.hydra is not None:
            self.hydra.initialize()
        # make the linear module orthogonal initialization
        self.first_layer.initialize( method='orthogonal' )
        if self.resnet is not None:
            self.resnet.initialize( layer_kwargs=dict( method=method ) )
        self.final_layer[0].initialize()
        return self
    
    def weight_regularization( self, ridge=None, lasso=None, smooth_lasso=None, logdet=None ):
        reg = 0
        if self.hydra is not None:
            reg = self.hydra.weight_regularization( ridge=ridge )
        reg += self.first_layer.weight_regularization( ridge=ridge, lasso=lasso, smooth_lasso=smooth_lasso,
                                                       logdet=logdet )
        if self.resnet is not None:
            reg += self.resnet.weight_regularization(
                layer_kwargs=dict( ridge=ridge, lasso=lasso, smooth_lasso=smooth_lasso, logdet=logdet ),
                gate_ridge=ridge, gate_lasso=lasso, gate_smooth_lasso=smooth_lasso,
            )
        reg += self.final_layer.weight_regularization( ridge=ridge, lasso=lasso, smooth_lasso=smooth_lasso )
        return reg
    
    def trg_loss( self, T, S, X, Y, W, mask=None, ):
        ''' the training loss composed of error + regularizations '''
        self.train( True )
        loss_hps = self.loss_hps
        # build up the regularization terms
        act_reg, Yhat = self.activation_regularization( X, kapow=loss_hps.kapow, tv=loss_hps.tv,
                                                        curvature=loss_hps.curvature )
        weight_reg = self.weight_regularization( ridge=loss_hps.ridge, lasso=loss_hps.lasso,
                                                 smooth_lasso=loss_hps.smooth_lasso, logdet=loss_hps.logdet )
        lip_reg, _ = self.lipchitz_regularization( X=X, penalty=loss_hps.lipschitz, lr=0.5, radius=1e-2,
                                                   num_its=5, tensorboard=self.monitor )
        total_reg = act_reg + weight_reg + lip_reg
        # if a mask is not provided cut out priming data from this. Otherwise the mask should handle it
        if mask is None:
            Yhat, Y, W = Yhat[loss_hps.priming:], Y[loss_hps.priming:], W[loss_hps.priming:]
        # compute the error
        if loss_hps.error == 'mse':
            err = self._mse( Yhat=Yhat, Y=Y, W=W, mask=mask )
        elif loss_hps.error == 'sr':
            err = self._psr( Yhat=Yhat, Y=Y, W=W, mask=mask )
        Wsum = W.sum() if mask is None else W[mask].sum()
        # --- logging stuff---
        if not hasattr( self, '_trg_it' ):
            self._trg_it = 0
        it = self._trg_it
        if it % loss_hps.patience == 0:
            try:
                mx.tensorboard.logger.add_scalar( f'TrgLoss/mse', self._mse( Yhat, Y, W, mask=mask ) , it )
                mx.tensorboard.logger.add_scalar( f'TrgLoss/psr', self._psr( Yhat, Y, W, mask=mask ) * np.sqrt( 252 ),
                                                  it )
                mx.tensorboard.logger.add_scalars( f'TrgLoss/regs', dict( total=total_reg, weight=weight_reg,
                                                                          act=act_reg, lip=lip_reg ), it )
                mx.tensorboard.logger.add_scalars( f'TrgLoss/sizes', dict( Wsum=Wsum, masked_Wsum=W[mask].sum() ), it )
            except Exception as e:
                self.warn( f'tensorboard failure {e} \n SKIPPING' )
        self._trg_it += 1
        return err + total_reg
    
    def val_loss( self, T, S, X, Y, W, mask=None ):
        self.train( False )
        loss_hps = self.loss_hps
        Yhat = self( X )
        # if a mask is not provided cut out priming data from this. Otherwise the mask should handle it
        if mask is None:
            Yhat, Y, W = Yhat[loss_hps.priming:], Y[loss_hps.priming:], W[loss_hps.priming:]
        # compute the error
        if loss_hps.error == 'mse':
            err = self._mse( Yhat=Yhat, Y=Y, W=W, mask=mask )
        elif loss_hps.error == 'sr':
            err = self._psr( Yhat=Yhat, Y=Y, W=W, mask=mask )
        if not hasattr( self, '_val_it' ):
            self._val_it = 0
        it = self._val_it
        if it % loss_hps.patience == 0:
            mx.tensorboard.logger.add_scalar( f'ValLoss/mse',  self._mse( Yhat, Y, W, mask=mask ), it )
            mx.tensorboard.logger.add_scalar( f'ValLoss/psr', self._psr( Yhat, Y, W, mask=mask ) * np.sqrt( 252 ), it )
            mx.tensorboard.logger.add_scalars( f'ValLoss/sizes', dict( Wsum=W.sum(), masked_Wsum=W[mask].sum() ), it )
        self._val_it += 1
        return err
