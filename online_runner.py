'''
The online runner operator shared by many notebooks in this directory
'''
from pylab import *
from madmax.api import *

from madmax.algos.online import pca as mxpca;
from madmax.base.operator import Operator
from madmax.operators import comparator as mxc;
mxtr = mxtr.reload()
mx.online = mx.online.reload()
reload(mxpca)
reload(mxpca)


def online_pipeline( data, hps, memoize=True, name=None ):
    '''
    run the online learning pipeline in staged manner end to end with staged checkpoits and baselining logic
    
    '''
    features = hps.features
    targets = hps.targets
    start, end = pd.to_datetime( hps.start ), pd.to_datetime( hps.end )
    # ---- do the online pca if provided ---
    if hps.pca is not None:
        # set the start and end date
        hps.pca.start = online_hps.start
        hps.pca.end = online_hps.end
        pca_runner = OnlinePCARunner( start=hps.start, end=hps.end, hps=hps.pca, weight=hps.weight )
        pca_runner.set( memoize=memoize, data=data, features=hps.pca.targets )
        with pca_runner.timer( 'running ...' ):
            pca_runner.run()
        # plot the pca results now
        for k, v in pca_runner.output_meta.items():
            disp( k, h=2 )
            disp( v )
        # convert all the output features into factors and eigenvalues
        pca_factors = mx.Features( '/pca_factor_.*/' ).find( pca_runner.output_features )
        pca_eigens = mx.Features( '/pca_eigval_.*/' ).find( pca_runner.output_features )
        # use the the factors as feautres
        if hps.pca.factors_as_features:
            features += pca_factors
        # use the eigens as features
        if hps.pca.eigens_as_features:
            features += pca_eigens
        # use factors as ortho targets
        if hps.pca.factors_as_ortho:
            hps.online.ortho_features += pca_factors
        # adjust the start dates of the online_runner to account for pca priming
        start += pd.to_timedelta( 60, 'D' )
        data = pca_runner
        
    # -- feature preprocessing ----
    setup = PreprocessFeatures(
        targets=targets, start=start, end=end, symbol=hps.preprocess.symbol, zscore_target=hps.preprocess.zscore_target,
        winsorize_target=hps.preprocess.winsorize_target, winsorize_features=hps.preprocess.winsorize_features,
        demean_features=hps.preprocess.demean_features, normalize_features=hps.preprocess.normalize_features,
    )
    setup.set( memoize=memoize, data=data, features=features )
    with setup.timer( 'running preprocess steps' ):
        setup()
    features = setup.output_features
    targets = setup.output_meta['targets']
    data = setup
    # -- online runner ----
    online_runner = OnlineRunner( targets=targets, weight=hps.weight, hps=hps.online, name=name,
                                  original_features=setup.output_meta['original_features'], )
    online_runner.set( memoize=memoize, data=data, features=features, verbose=True )
    return online_runner


@mx.operatorize( memoize=True, consumes_features=True, produces_features=True, produces_meta=True )
def PreprocessFeatures( da, features, targets, start, end, winsorize_target, zscore_target,
                        winsorize_features, demean_features, normalize_features, symbol=None ):
    '''winsorize and clean up the features'''
    logger = mx.Logger( 'OnlineSetup' )
    if start:
        da = da.loc[start:]
    if end:
        da = da.loc[:end]
    # make sure you don't squeeze out this dimension !
    if symbol:
        da = da.loc[:, mx.make_iterable( symbol ), :]
    # clean up any dangling forecasts that will cause name collisonss later
    bad_features = da.find_mx( ['/leadrtn_adj_30min.*hat_xs_winsorize/', '/leadrtn_adj_30min.*hat_xs_zscore/',
                                '/leadrtn_adj_30min.*hat_xs_winsorize_xs_zscore/'] )
    if bad_features:
        logger.warn( 'Deleting bad features in input data ', bad_features )
        da = da.drop_coords( features=bad_features )
    original_features = da.features.values.tolist()
    features = make_iterable( features )
    targets = make_iterable( targets )
    if winsorize_target:
        with logger.timer( 'pre normalizing the targets returns' ):
            prewins = mx.transforms.cross_sectional( sid='symbol' ).winsorize( quantiles=winsorize_target )
            da = prewins( da, features=targets )
            targets = prewins.output_features
    if zscore_target:
        prenorm = mx.transforms.cross_sectional( sid='symbol' ).zscore(  )
        da = prenorm( da, features=targets )
        targets = prenorm.output_features
    if winsorize_features:
        with logger.timer( '# pre-winsorize the input features' ):
            wins = mx.transforms.batch( sid='symbol' ).winsorize( quantiles=winsorize_features )
            da = wins( da, features=features )
            features = wins.output_features
    if demean_features:
        with logger.timer( '# batch-demean the input features' ):
            batch_demean = mx.transforms.batch( sid=None ).demean()
            da = batch_demean( da, features=features )
            features = batch_demean.output_features
    if normalize_features:
        with logger.timer( '# batch-normalize the input features' ):
            batch_normalize = mx.transforms.batch( sid=None ).normalize()
            da = batch_normalize( da, features=features )
            features = batch_normalize.output_features
    return da, features, dict( targets=targets, original_features=original_features )


class PreTransformer( tr.nn.Module ):
    '''manage the pre-transformation in the online pipeline'''
    
    def __init__( self, pre_normalize, pre_clip, quadratic, ema_halflifes, post_normalize, post_clip ):
        super().__init__()
        self.pre_normalize = pre_normalize
        self.pre_clip = pre_clip
        self.quadratic = quadratic
        self.ema_halflifes = ema_halflifes
        self.post_normalize = post_normalize
        self.post_clip = post_clip
        if self.pre_normalize:
            self._pre = mx.online.OnlineNorm( demean=True, normalize=True, clip=self.pre_clip, halflife='xw' )
        if self.quadratic:
            self._quad = mx.online.OnlineFeatures( transform='quadratic' )
        if self.ema_halflifes:
            self._ema = mx.online.OnlineEma( halflifes=self.ema_halflifes )
        if self.post_normalize:
            self._post = mx.online.OnlineNorm( demean=True, normalize=True, clip=self.post_clip, halflife='xw' )
        self.initialize()
    
    def initialize( self ):
        for chld in self.children():
            chld.initialize()
        return self
    
    def forward( self, X_t, W_t ):
        '''one pre transformation step'''
        if self.pre_normalize:
            X_t = self._pre( X_t, W_t=W_t )
        if self.quadratic:
            X_t = self._quad( X_t, W_t=W_t )
        if self.ema_halflifes:
            X_t = self._ema( X_t, W_t=W_t )
        if self.post_normalize:
            X_t = self._post( X_t, W_t=W_t )
        return X_t


class OnlineRunner( Operator ):
    ''' run one online learning configuration end to end with profiling and name management'''
    
    def __init__( self, targets, original_features, weight, hps, name=None, dtype=tr.float32, device='cuda' ):
        self.targets = make_iterable( targets )
        self.original_features = make_iterable( original_features )
        self.hps = hps.clone()
        self.weight = weight
        self.name = name
        self.dtype = dtype
        self.device = device
        super().__init__()
        if hps.pre_transformer:
            self._pre_transformer = PreTransformer(
                pre_normalize=hps.pre_transformer.pre_normalize, pre_clip=hps.pre_transformer.pre_clip,
                quadratic=hps.pre_transformer.quadratic, ema_halflifes=hps.pre_transformer.ema_halflifes,
                post_normalize=hps.pre_transformer.post_normalize, post_clip=hps.pre_transformer.post_clip
            )
        else:
            self._pre_transformer = None
        if hps.post_transformer and hps.post_transformer.winsorize:
            # do cross sectional post winsorization along N
            self._post_transformer = mxol.OnlineWinsorize( winsorize=hps.post_transformer.winsorize )
        else:
            self._post_transformer = None
    
    def apply( self, da, features ):
        '''run the transformer'''
        hps = self.hps
        online_rnn = mx.online.OnlineRNN.operator(
            targets=self.targets, weight=self.weight, ortho_features=hps.ortho_features, batch_size=hps.batch_size,
            dtype=self.dtype, device=self.device, verbose=self.verbose, lookahead=hps.lookahead, ridge=hps.ridge,
            kappa=hps.kappa, algo=hps.algo, halflife_1=hps.hl_1, halflife_2=hps.hl_2, lr=hps.lr, tv=hps.tv,
            ortho=hps.ortho, use_omega=hps.use_omega, omega_lr=hps.omega_lr, omega_ridge=hps.omega_ridge,
            omega_shift=hps.omega_shift, halflife_ortho=hps.ortho_hl, halflife_kappa=hps.halflife_kappa,
            pre_transformer=self._pre_transformer, post_transformer=self._post_transformer
        )
        for _ in range( 5 ):
            gc.collect();
            tr.cuda.empty_cache()
        da = online_rnn( data=da, features=features )
        output_features = online_rnn.output_features
        del online_rnn
        # clean the dataframe of any garbage created upto this point
        da = da[500:].loc[:, :, self.original_features + output_features]
        for _ in range( 5 ):
            gc.collect();
            tr.cuda.empty_cache()
        with self.timer( 'post winsorizations' ):
            wins = mx.transforms.cross_sectional( sid='symbol' ).winsorize( quantiles=0.05 )
            da = wins( da, features=output_features )
        with self.timer( 'post zscore' ):
            zscore = mx.transforms.cross_sectional( sid='symbol' ).zscore( weight='weight' )
            da = zscore( da, features=wins.output_features )
            output_features += zscore.output_features
        # rename the forecasts to be unique
        suffix = self.name if self.name else self.hash()
        rename_map = OrderedDict( [(ft, f'{ft}_{suffix}') for ft in output_features] )
        da = da.rename_coords( features=rename_map )
        output_features = list( rename_map.values() )
        with self.timer( 'profiling ' ):
            profiler = mx.FeatureProfiler( returns=hps.regression_targets, exposures=hps.regression_exposures,
                                           weight='weight', transforms=None, winsorize=None, xs=None, lags=None,
                                           autocorr=[1, 2, 13], remove_intercept=False, )
            profiler.data = da
            profiler.features = output_features
            profiler.run()
        output_meta = dict( profiler=profiler.output_meta, comparator=None )
        return da, output_features, output_meta


@mx.operatorize( produces_features=True, produces_meta=True )
def OnlinePCARunner( data, features, start, hps, weight  ):
    '''online pca runner'''
    if start:
        data = data.loc[start:]
    if end:
        data = data.loc[:end]
    for _ in range( 5 ):
        gc.collect();
        tr.cuda.empty_cache()
    mx.seed( 17 )
    
    # the online pca operator
    pca = mxpca.OnlinePCA.operator( weight=weight, ortho_features=hps.ortho_features, batch_size=hps.batch_size,
                                    halflife=hps.halflife, rank=hps.rank, kappa=hps.kappa, ortho=hps.ortho, tv=hps.tv,
                                    ortho_halflife=hps.ortho_halflife, ridge=hps.ridge, verbose=True, metadata=True, )
    data = pca( data=data, features=features )
    data = data.rename_coords( features={ x: f'{x}_kappa.{hps.kappa}_hl.{hps.halflife}' for x in pca.output_features } )
    output_factors = mx.Features( '/pca_factor.*/' ).find( data )
    output_eigens = mx.Features( '/pca_eigval.*/' ).find( data )
    
    # build output meta for returning
    results = pca.output_meta
    output_meta = dict( profiles=results['profiles'] )
    
    K = hps.rank
    P = len( features )
    R = len( hps.ortho_features )
    
    pnl = results['pnls']
    fig, axs = subplots( P, K, figsize=(K * 4, P * 4) )
    axs = np.atleast_2d( axs )
    for k, fac in enumerate( pnl.factors.values ):
        for p, tgt in enumerate( pnl.targets.values ):
            ax = axs[p][k]
            ax.plot( pnl[:, k, p].cumsum().to_series() / pnl[:, k, p].sum().item() )
            ax.set_xlabel( fac, fontsize=11 )
            ax.set_ylabel( tgt, fontsize=11 )
            ax.set_title( f'{fac} x {tgt}' )
    fig.suptitle( 'pnl', fontsize=16 )
    output_meta['pnl'] = mx.fig2format( fig )
    
    ic = results['ics']
    fig, axs = subplots( P, K, figsize=(K * 4, P * 4), sharex=True, sharey=True )
    axs = np.atleast_2d( axs )
    for k, fac in enumerate( ic.factors.values ):
        for p, tgt in enumerate( ic.targets.values ):
            ax = axs[p][k]
            ax.plot( ic[:, k, p].abs().to_series() * 100, '--k', alpha=0.1 )
            ax.plot( ic[:, k, p].to_series().rolling( 100 ).mean() * 100, '-k', alpha=0.8 )
            ax.set_xlabel( fac, fontsize=11 )
            ax.set_ylabel( tgt, fontsize=11 )
            ax.set_title( f'{fac} x {tgt}' )
    fig.suptitle( 'ics', fontsize=16 )
    output_meta['ics'] = mx.fig2format( fig )
    
    oic = results['oics']
    if oic is not None:
        fig, axs = subplots( R, K, figsize=(K * 4, R * 4) )
        axs = np.atleast_2d( axs )
        for k, fac in enumerate( oic.factors.values ):
            for r, ortho in enumerate( oic.orthos.values ):
                ax = axs[r][k]
                ax.plot( oic[:, k, r].to_series() * 100, '--k', alpha=0.1 )
                ax.plot( oic[:, k, r].to_series().rolling( 100 ).mean() * 100, '-k', alpha=0.8 )
                ax.set_xlabel( fac, fontsize=11 )
                ax.set_ylabel( ortho, fontsize=11 )
                ax.set_title( f'{fac} x {ortho}' )
        fig.suptitle( 'orthog-ics', fontsize=16 )
        output_meta['oics'] = mx.fig2format( fig )
    
    ac1 = results['ac1s']
    E = data.loc[:, 'SPY', output_eigens]
    fig, axs = subplots( 3, K, figsize=(K * 4, 15) )
    for k, fac in enumerate( pnl.factors.values ):
        ax = axs[0][k]
        ax.plot( ac1[:, k].to_series() * 100, '--k', alpha=0.1 )
        ax.plot( ac1[:, k].to_series().rolling( 100 ).mean() * 100, '-k', alpha=0.8 )
        ax.set_ylabel( 'ac1', fontsize=11 )
        ax.set_title( f'{fac} ac1' )
        ax = axs[1][k]
        ax.set_ylabel( 'AC1', fontsize=11 )
        ax.plot( E[:, k].to_series() * 100, '--k', alpha=0.1 )
        ax.plot( E[:, k].to_series().rolling( 100 ).mean() * 100, '-k', alpha=0.8 )
        ax.set_title( f'{fac} eig' )
        ax.set_ylabel( 'eig', fontsize=11 )
    # ax = axs[2][k]
    # ax.plot( X_norm[:, k].to_series(), '--k', alpha=0.1 )
    # ax.plot( X_norm[:, k].to_series().rolling( 100 ).mean() * 100, '-k', alpha=0.8 )
    # ax.set_title( f'{fac} norm' )
    # ax.set_ylabel( 'norm', fontsize=11 )
    fig.suptitle( 'ac1, eigvals and norm', fontsize=16 )
    output_meta['facts'] = mx.fig2format( fig )
    return data, output_factors + output_eigens, output_meta



# measure the performance of the individual and aggregate forecasts

@mx.operatorize( consumes_features=True, produces_features=True, produces_meta=True, memoize=True )
def EnsembleForecasts( data, features, ensemble_forecast, weight='valid_30min',
                       targets=['leadrtn_adj_30min', 'leadrtn_adj_1hr'],
                       exposures=['valid_30min', 'lagrtn_adj_30min_1170.ew_beta', 'lagrtn_adj_30min'] ):
    '''combines a bunch of boosted forecasts into one equally weighted forecast'''
    # drop all intermediate forecats
    data = data.drop_mx( [  '/leadrtn_adj_30min_hat_OnlineRunner.*/',
                            '/leadrtn_adj_30min_hat_xs_winsorize_OnlineRunner.*/',
                            '/leadrtn_adj_30min_hat_xs_zscore_OnlineRunner.*/' ] )
    # fcst-sd weighted average of the constitudent forecasts
    fcst_sd = data.loc[:, :, features].std( dim=data.dims[:-1] )
    data = data.assign_features( **{ ensemble_forecast: (data.loc[:, :, features] / fcst_sd).mean( dim=data.dims[-1] ) } )
    output_features = features + [ensemble_forecast]
    # profile the individual forecasts
    profiler = mx.FeatureProfiler(
        returns=targets, exposures=exposures, weight=weight, transforms=None, winsorize=None,
        xs=None, lags=[1, 2], autocorr=[1, 2, 13], remove_intercept=False,
    ).set( verbose=False, memoize=False, data=data, features=output_features )
    profiler.run()
    comparator = mxc.ForecastComparator( targets=targets, weight=weight, winlen=None )
    comparator.data = data
    comparator.features = output_features
    comparator.run()
    output_meta = dict( profiler=profiler.output_meta, comparator=comparator.output_meta )
    return data, output_features, output_meta



