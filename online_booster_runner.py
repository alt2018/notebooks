'''
OnlineBooster pipeline
'''
from pylab import *
from madmax.api import *

mxtr = mxtr.reload()

mx.online = mx.online.reload()


def run_pipeline( data, hps, prev_baseline, memoize=True, device='cuda', dtype=tr.float32 ):
    '''run the OnlineBooster pipeline'''
    # run the preprocessing pipeline
    preprocess = Preprocess(
        targets=hps.targets, start=hps.start, end=hps.end, weight=hps.weight,
        winsorize_target=hps.preprocess.winsorize_target, zscore_target=hps.preprocess.zscore_target,
        winsorize_weight=hps.preprocess.winsorize_weight, normalize_features=hps.preprocess.normalize_features,
        winsorize_features=hps.preprocess.winsorize_features, demean_features=hps.preprocess.demean_features,
    ).set( memoize=memoize, data=data, features=hps.features )
    with mx.timer( f'running {preprocess.hash()}' ):
        preprocess.run()
    features = preprocess.output_features
    targets = preprocess.output_meta['targets']
    weight = preprocess.output_meta['weight']
    
    # build an online booster or regular online operator
    if hps.boosters:
        learner = build_online_booster(
            hps, lookahead=hps.lookahead, num_features=len( features ), operator=True, targets=targets, weight=weight,
            ortho_features=hps.ortho_features, batch_size=hps.batch_size, dtype=dtype, device=device, verbose=True
        )
    # build a regular online rnn
    else:
        learner = build_online_rnn(
            hps=hps.ensembler, lookahead=hps.lookahead, num_features=len( features ), operator=True, targets=targets,
            weight=weight, ortho_features=hps.ortho_features, batch_size=hps.batch_size, dtype=dtype, device=device,
            verbose=True,
        )
    learner.set( memoize=memoize, data=preprocess, features=features )
    with mx.timer( f'Running {learner.hash()}' ):
        learner.run()
    # show runtime statistics
    disp( learner.output_meta['profiles'] )
    # do the post processing
    profiler = profile_and_cleanup(
        name=learner.hash(), prev_baseline=prev_baseline, targets=hps.profiling.targets,
        exposures=hps.profiling.exposures, weight=hps.profiling.weight, xsw=hps.profiling.xsw, xsz=hps.profiling.xsz,
        clip=hps.profiling.clip, priming=500
    ).set( memoize=memoize, data=learner, features=learner.output_features )
    return profiler


@mx.operatorize( memoize=True, consumes_features=True, produces_features=True, produces_meta=True )
def Preprocess( da, features, targets, start, end, weight, winsorize_target, zscore_target, winsorize_weight,
                winsorize_features, demean_features, normalize_features, symbol=None ):
    '''winsorize and clean up the features'''
    logger = mx.Logger( 'Preprocess' )
    if start:
        da = da.loc[start:]
    if end:
        da = da.loc[:end]
    # make sure you don't squeeze out this dimension !
    if symbol:
        da = da.loc[:, mx.make_iterable( symbol ), :]
    # clean up any dangling forecasts that will cause name collisonss later
    bad_features = da.find_mx( ['/leadrtn_adj.*hat_xs_winsorize/', '/leadrtn_adj.*hat_xs_zscore/',
                                '/leadrtn_adj.*hat_xs_winsorize_xs_zscore/'] )
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
        prenorm = mx.transforms.cross_sectional( sid='symbol' ).zscore( weight=weight )
        da = prenorm( da, features=targets )
        targets = prenorm.output_features
    if winsorize_features:
        with logger.timer( '# pre-winsorize the input features' ):
            wins = mx.transforms.batch( sid='symbol' ).winsorize( quantiles=winsorize_features )
            da = wins( da, features=features )
            features = wins.output_features
    if demean_features:
        with logger.timer( '# batch-demean the input features' ):
            batch_demean = mx.transforms.batch( sid=None ).demean( weight=weight )
            da = batch_demean( da, features=features )
            features = batch_demean.output_features
    if normalize_features:
        with logger.timer( '# batch-normalize the input features' ):
            batch_normalize = mx.transforms.batch( sid=None ).normalize( weight=weight )
            da = batch_normalize( da, features=features )
            features = batch_normalize.output_features
    if winsorize_weight:
        with logger.timer( 'normalizing weights to be well posed' ):
            wins = mx.transforms.cross_sectional( sid='symbol' ).winsorize( quantiles=0.05 )
            da = wins( da, features=weight )
            weight = wins.output_features[0]
    return da, features, dict( targets=targets, original_features=original_features, weight=weight )


def build_transformer( hps, num_features ):
    transforms = []
    assert len( set( hps.keys() ).difference(
        { 'subset', 'winsorize', 'pre_normalize', 'hydra', 'transform', 'post_normalize', 'ema_halflifes' }
    ) ) == 0, 'bad arguments'
    if getattr( hps, 'winsorize', False ):
        transforms.append( mx.online.OnlineWinsorize( winsorize=hps.winsorize ) )
    if getattr( hps, 'pre_normalize', False ):
        transforms.append( mx.online.OnlineNorm( demean=True, normalize=True, clip=hps.pre_normalize.clip,
                                                 halflife=hps.pre_normalize.halflife ) )
    if getattr( hps, 'subset', False ):
        transforms.append(
            mx.online.OnlineSubset( frac=hps.subset.frac, num_features=num_features, seed=hps.subset.seed )
        )
    if getattr( hps, 'hydra', False ):
        hydra = mx.online.Hydra()
        transforms.append( hydra )
        # account for the new features that hydra will create
        num_features = num_features * (len( hydra.activations ) + 1)
    if getattr( hps, 'transform', False ):
        transforms.append( mx.online.OnlinePolynomial( order=hps.transform ) )
    if getattr( hps, 'ema_halflifes', False ):
        transforms.append( mx.online.OnlineEma( halflifes=hps.ema_halflifes ) )
    if getattr( hps, 'post_normalize', False ):
        transforms.append( mx.online.OnlineNorm( demean=True, normalize=True, clip=hps.post_normalize.clip,
                                                    halflife=hps.post_normalize.halflife ) )
    return mx.online.OnlineSequential( *transforms )


def build_online_rnn( hps, lookahead, num_features, operator=False, targets=None, weight=None, ortho_features=None,
                      batch_size=None, dtype=tr.float32, device='cuda', verbose=True ):
    '''build an olnine rnn (module or operator) from the given set of hyperparameters'''
    # ubild out the pre and post transformer
    pre_transformer = (build_transformer( hps.pre_transformer, num_features=num_features )
                       if hps.pre_transformer else None)
    post_transformer = build_transformer( hps.post_transformer, num_features=None ) if hps.post_transformer else None
    if operator:
        rnn = mx.online.OnlineRNN.operator(
            targets=targets, weight=weight, ortho_features=ortho_features, batch_size=batch_size, dtype=dtype,
            device=device, verbose=verbose, loss=getattr( hps, 'loss', 'psr' ),
            lookahead=lookahead, algo=hps.algo, lr=hps.lr, pre_transformer=pre_transformer,
            post_transformer=post_transformer, halflife_1=hps.hl_1, halflife_2=hps.hl_2,
            halflife_polyak=hps.halflife_polyak,
            ridge=hps.ridge, tv=hps.tv, kappa=hps.kappa, halflife_kappa=hps.halflife_kappa, ortho=hps.ortho,
            halflife_ortho=hps.ortho_hl, halflife_algo=hps.halflife_algo,
            use_omega=hps.use_omega, omega_lr=hps.omega_lr, omega_ridge=hps.omega_ridge, omega_shift=hps.omega_shift,
        )
    else:
        rnn = mx.online.OnlineRNN(
            lookahead=lookahead, algo=hps.algo, lr=hps.lr, pre_transformer=pre_transformer,
            post_transformer=post_transformer, halflife_1=hps.hl_1, halflife_2=hps.hl_2,
            halflife_polyak=hps.halflife_polyak, loss=getattr( hps, 'loss', 'psr' ),
            ridge=hps.ridge, tv=hps.tv, kappa=hps.kappa, halflife_kappa=hps.halflife_kappa, ortho=hps.ortho,
            halflife_ortho=hps.ortho_hl, halflife_algo=hps.halflife_algo,
            use_omega=hps.use_omega, omega_lr=hps.omega_lr, omega_ridge=hps.omega_ridge, omega_shift=hps.omega_shift,
        )
        rnn.verbose = verbose
    return rnn


def build_online_booster( hps, lookahead, num_features, operator=False, targets=None, weight=None, ortho_features=None,
                        batch_size=None, dtype=tr.float32, device='cuda', verbose=True ):
    '''build a OnlineBooster operator or module'''
    # build the pre and post transformers
    pre_transformer = (build_transformer( hps.pre_transformer, num_features=num_features )
                       if hps.pre_transformer else None)
    post_transformer = build_transformer( hps.post_transformer, num_features=None ) if hps.post_transformer else None
    # build the booster rnns
    boosters = [build_online_rnn( bhps, lookahead=lookahead, num_features=num_features, operator=False, verbose=False )
                for bhps in hps.boosters]
    ensembler = build_online_rnn( hps.ensembler, lookahead=lookahead, num_features=num_features, operator=False,
                                  verbose=False )
    if operator:
        rnn = mx.online.OnlineBooster.operator(
            targets=targets, weight=weight, ortho_features=ortho_features, batch_size=batch_size, dtype=dtype,
            device=device, verbose=verbose, boosters=boosters, ensembler=ensembler, lookahead=lookahead,
            post_transformer=post_transformer, pre_transformer=pre_transformer
        )
    else:
        rnn = mx.online.OnlineBooster( lookahead=lookahead, boosters=boosters, ensembler=ensembler,
                                  post_transformer=post_transformer, pre_transformer=pre_transformer )
        rnn.verbose = verbose
    return rnn


@mx.operatorize( produces_meta=True, produces_features=True, consumes_features=True )
def profile_and_cleanup( data, features, name, targets, exposures, weight, prev_baseline, xsw=False, xsz=False,
                         clip=None, priming=500 ):
    '''profile the results of a dataset and baseline if needed '''
    logger = mx.Logger( 'profile_and_baseline', verbose=True )
    # drop everything except stuff of interest
    keep_features = sorted( set(
        features + make_iterable( targets ) + make_iterable( exposures ) + make_iterable( weight )
    ) )
    # clean up an garbage features and drop some shit for priming
    data = data[priming:].loc[:, :, keep_features]
    if xsw:
        with logger.timer( 'winsorizing predictions' ):
            wins = mx.transforms.cross_sectional( sid='symbol' ).winsorize( quantiles=0.05 )
            data = wins( data=data, features=features )
            features = wins.output_features
    if xsz:
        with logger.timer( 'zscoring predictions' ):
            xsz = mx.transforms.cross_sectional( sid='symbol' ).zscore( weight=weight )
            data = xsz( data=data, features=features )
            features = xsz.output_features
    if clip:
        with logger.timer( 'clipping' ):
            data = xa.concat( [data.drop_mx( features ), data.loc[:, :, features].clip( -clip, clip )], dim='features' )
    # rename the coordinates
    output_features = [f'{ft}_{name}' for ft in features]
    data = data.rename_coords( features=dict( zip( features, output_features ) ) )
    with logger.timer( 'profiling ' ):
        profiler = mx.FeatureProfiler( returns=targets, exposures=exposures, weight=weight, transforms=None,
                                       winsorize=None, xs=None, lags=None, autocorr=[1, 2, 13], remove_intercept=False )
        profiler.run( data, output_features )
    output_meta = dict( profiler=profiler.output_meta, current_forecasts=output_features )
    # merge with the other baseline if present
    if prev_baseline is not None:
        baseline = mx.Baseline.load( prev_baseline )
        baseline_da = baseline()
        keep_features = sorted( set( data.features.values ) - set( baseline_da.features.values ) )
        data = xa.concat( [baseline_da, data.loc[:, :, keep_features]], dim='features' )
        # chain the baseline forecasts to the current guy
        output_features = baseline.output_features + output_features
    return data, output_features, output_meta


@mx.operatorize( produces_features=True, produces_meta=True )
def OnlinePCARunner( data, features, weight, rank, kappa, ridge, tv, halflife, ortho_features, ortho, ortho_halflife,
                     suffix, start, end, batch_size, dtype=tr.float32, device='cuda' ):
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
    pca = mxpca.OnlinePCA.operator( weight=weight, ortho_features=ortho_features, batch_size=batch_size,
                                    halflife=halflife, rank=rank, kappa=kappa, ortho=ortho, tv=tv,
                                    ortho_halflife=ortho_halflife, ridge=ridge, verbose=True, metadata=True,
                                    dtype=dtype, device=device )
    data = pca( data=data, features=features )
    data = data.rename_coords( features={ x: f'{x}{suffix}' for x in pca.output_features } )
    output_factors = mx.Features( '/pca_factor.*/' ).find( data )
    output_eigens = mx.Features( '/pca_eigval.*/' ).find( data )
    
    # build output meta for returning
    results = pca.output_meta
    output_meta = dict( profiles=results['profiles'], output_factors=output_factors, output_eigens=output_eigens )
    
    # --- do plotting ---
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
