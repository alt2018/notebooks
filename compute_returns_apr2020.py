'''
Compute returns for the 2020-Apr data

goes with etf_technical_2020.05.05.v1.ipynb notebook
'''

from pylab import *
from madmax.api import *
import gc


@mx.operatorize( consumes_features=False, produces_features=False, produces_meta=True, memoize='d' )
def compute_returns_betas_and_residuals( da, start, end,  emv_halflifes, dollar_volume_window, resid_halflife,
                                         beta_halflifes, volscale=False, eps=1e-3, plots=True ):
    '''
    compute log-returns, log-dollar-volume, betas, residual returns and clean up the data
    
    use the `close` price to compute leading and lagging returns
    
    # HERO has a massive returns spike in 2015.11.09 - which throws off residualization quite a bit.
    # one option is to truncate it very aggressively when doing beta computation and residualization
    # second is to use more agreesive weights
    
    '''
    logger = mx.Logger( 'compute_returns_betas_and_residuals', verbose=True )
    _orig_shape = da.shape
    if start:
        da = da.loc[start:]
    if end:
        da = da.loc[:end]
    # remove flash crash dates !
    flash_crash = (pd.to_datetime( da.time.values ).date == pd.to_datetime( '20150824' ))
    logger.info( f'Will thow away {flash_crash.sum()} rows because of flashcrash' )
    da = da[~flash_crash]
    # restrict the data to market hours
    market_times = [dt.time( 10, 0 ) <= t <= dt.time( 16, 0 ) for t in pd.to_datetime( da.time.values ).time]
    da = da[market_times]
    logger.info( 'original', _orig_shape, 'restricted', da.shape )
    
    # names of primary columns
    leadrtn = 'leadrtn_adj_30min'
    lagrtn = 'lagrtn_adj_30min'
    log_dollar_volume = 'log10_dollar_volume_30min'
    mktrtn = 'mkt_lagrtn_30min'
    weight = 'weight'
    
    with logger.timer( " measure universe coverage and intervals per day" ):
        valid_count = da.loc[:, :, 'valid'].sum( dim='symbol' ).to_series()
        ints_every_day = da.time.to_dataframe().assign( date=lambda df: df.index.date ).groupby( 'date' ).count()
        assert all( ints_every_day == 13 ), 'not 13 intervals every day - fix it'
    
    with logger.timer( 'adding calendar features' ):
        cf_df = mx.calendar_features( da.time.to_dataframe() ).drop_mx( 'time' )
        cf_da = cf_df.to_dataarray_mx()
        da = da.assign_features( cal_features=cf_da.rename( { cf_da.dims[-1]: da.dims[-1] } ) )
    
    with logger.timer( ' compute returns' ):
        for sfx, periods in logger.pbar( [('30min', 1), ('1hr', 2), ('1day', 13)] ):
            # fjnote - a tighter clamp on max returns (30% in 30mins)
            lagrtn_computer = mx.returns( sid='symbol' ).lagging_da( periods=periods, max_gap=13 * 3,
                                                                     max_rtn=periods * 0.2 )
            leadrtn_computer = mx.returns( sid='symbol' ).leading_da( periods=periods, max_gap=13 * 3,
                                                                      max_rtn=periods * 0.2 )
            # FJNOTE - note that I'm using close prices for this
            da = lagrtn_computer( da, features='close_adj_30min' )
            da = leadrtn_computer( da, features='close_adj_30min' )
            da = da.rename_coords( features={ lagrtn_computer.output_features[0]: f'lagrtn_adj_{sfx}',
                                              leadrtn_computer.output_features[0]: f'leadrtn_adj_{sfx}' } )
    
    with logger.timer( 'adding volume, weight and vol features' ):
        da = da.assign_features(
            log10_dollar_volume_30min=da.loc[:, :, 'dollar_volume_unadj_30min'].log10(),
            mkt_lagrtn_30min=da.loc[:, 'SPY', lagrtn],
            mkt_leadrtn_30min=da.loc[:, 'SPY', leadrtn],
            mkt_leadrtn_1day=da.loc[:, 'SPY', 'leadrtn_adj_1day'],
        )
        da = da.assign_features(
            delta_rtn_30min=((da.loc[:, :, 'close_unadj_30min'] - da.loc[:, :, 'open_unadj_30min'])
                             / da.loc[:, :, 'open_unadj_30min']),
            atr_30min=((da.loc[:, :, 'high_unadj_30min'] - da.loc[:, :, 'low_unadj_30min'])
                       / da.loc[:, :, 'close_unadj_30min']),
            delta_log10_dollar_volume_30min=da.loc[:, :, 'log10_dollar_volume_30min'].diff( dim='time', n=1 ),
            avg_dollar_volume=(da.loc[:, :, 'dollar_volume_unadj_30min']
                                     .rolling( time=dollar_volume_window, min_periods=1 ).mean()),
        )
        # weight is sqrt avg log dollar volume of a sid normalized by a rolling average of the cross sectional
        # median of log dollar volume
        weight_denom = (da.loc[:, :, 'avg_dollar_volume'].sqrt()
                        .median( dim='symbol' )
                        .rolling( time=dollar_volume_window, min_periods=1 )
                        .mean())
        da = da.assign_features( weight=da.loc[:, :, 'avg_dollar_volume'].sqrt() / weight_denom )
        # compute raw returns and volume volatility
        ems = mx.transforms.exponential( sid='symbol', halflifes=emv_halflifes ).sd( weight=weight )
        da = ems( da, features=[log_dollar_volume, lagrtn, mktrtn, 'delta_log10_dollar_volume_30min'] )
        
    gc.collect()
    
    if volscale:
        with logger.timer( 'volscaling the returns and volumes' ):
            xws = mx.transforms.expanding( sid='symbol' ).sd( weight=weight )
            da = xws( da, features=[lagrtn, log_dollar_volume] )
            da = da.assign_features( **{
                f'{lagrtn}_volscale': da.loc[:, :, lagrtn] / (da.loc[:, :, f'{lagrtn}_xw_sd'] + eps),
                f'{leadrtn}_volscale': da.loc[:, :, leadrtn] / (da.loc[:, :, f'{lagrtn}_xw_sd'] + eps),
                f'{log_dollar_volume}_volscale': (da.loc[:, :, log_dollar_volume]
                                                  / (da.loc[:, :, f'{log_dollar_volume}_xw_sd'] + eps)),
            } )
            lagrtn = f'{lagrtn}_volscale'
            leadrtn = f'{leadrtn}_volscale'
            log_dollar_volume = f'{log_dollar_volume}_volscale'
    
    with logger.timer( 'computing market betas and market return ' ):
        beta_computer = mx.returns( sid='symbol' ).factor_loadings( factor_rtn=mktrtn, halflifes=beta_halflifes )
        beta_da = beta_computer( da, features=lagrtn )
        da = da.assign_features( betas=beta_da )
    
    with logger.timer( 'residualizing returns and volume for tilt and beta' ):
        updater = mx.LinearUpdater( loss='l2', ridge=1e-5, algo='direct', hl_1=resid_halflife,
                                    hl_2=10 * resid_halflife )
        residualizer = (mx.returns( sid='symbol' )
                        .residualize( response=[lagrtn, leadrtn, log_dollar_volume], weight='weight', updater=updater )
                        .set( verbose=True ))
        da_rsd = residualizer( da.fillna( 0 ), features=['valid'] + beta_computer.output_features )
        da = xa.concat( [da, da_rsd.loc[:, :, residualizer.output_features]], dim='features' )
    
    with logger.timer( 'computing residual volatility ' ):
        resid_ems = ems.clone()
        da = resid_ems( da, features=mx.Features( '/leadrtn.*/' ).drop( residualizer.output_features ) )
    
    if not plots:
        return da, []
    
    figs = []
    with logger.timer( 'plotting general stats' ):
        fig = figure( figsize=(12, 15) )
        subplot( 3, 2, 1 )
        plot( valid_count / da.shape[1], '.', alpha=0.5 )
        subplot( 3, 2, 2 )
        plot( ints_every_day, '.' )
        gca().set_title( 'num intervals per day', fontsize=12 )
        subplot( 3, 2, 3 )
        plot( da.loc['2018'].time, da.loc['2018', 'YINN', 'quarterofyear'], '--b' )
        plot( da.loc['2018'].time, da.loc['2018', 'YINN', 'monthofquarter'], '-k' )
        plot( da.loc['2018'].time, da.loc['2018', 'AGG', 'weekofmonth'], '--g' )
        gca().set_title( 'calendar features', fontsize=12 )
        subplot( 3, 2, 4 )
        plot( weight_denom.to_series() )
        gca().set_title( 'weight_denom', fontsize=12 )
        subplot( 3, 2, 5 )
        mkt_vol = mx.Features( '/mkt_lagrtn_30min.*/' ).find( ems.output_features )[0]
        plot( da.loc[:, 'SPY', mkt_vol].to_series() )
        gca().set_title( 'market volatility', fontsize=12 )
        fig.suptitle( 'general stats', fontsize=15 )
        figs.append( fig )
    
    with logger.timer( 'plotting returns results' ):
        fig = figure( figsize=(15, 15) )
        for i, sym in enumerate( random_sample( da.symbol.values, 16 ) ):
            ax = subplot( 4, 4, i + 1 )
            cp = (1 + da.loc[:, sym, 'lagrtn_adj_30min']).cumprod( dim='time' )
            ax.plot( da.loc[:, sym, 'close_adj_30min'] / da.loc[:, sym, 'close_adj_30min'][-1], cp / cp[-1], '.' )
            ax.set_title( sym )
            ax.set_xlabel( 'open_adj' )
            ax.set_ylabel( 'cum lag rtn' )
        fig.suptitle( 'cum lag rtn vs open', fontsize=15 )
        figs.append( fig )
        
        fig = figure( figsize=(15, 15) )
        for i, sym in enumerate( random_sample( da.symbol.values, 16 ) ):
            subplot( 4, 4, i + 1 )
            plot( (1 + da.loc[:, sym, 'lagrtn_adj_30min']).rolling( time=13 ).prod() - 1,
                  da.loc[:, sym, 'lagrtn_adj_1day'], '.b', alpha=0.2, label='30min vs 1day' )
            plot( (1 + da.loc[:, sym, 'lagrtn_adj_30min']).rolling( time=2 ).prod() - 1,
                  da.loc[:, sym, 'lagrtn_adj_1hr'], '.r', alpha=0.3, label='30min vs 1hrr' )
            gca().set_title( sym )
            gca().legend()
        fig.suptitle( 'lag returns 30min vs 1day vs 1hr', fontsize=15 )
        figs.append( fig )
    
    gc.collect()
    
    with logger.timer( 'plotting returns results - 2' ):
        fig = figure( figsize=(10, 10) )
        ax = subplot( 2, 2, 1 )
        ax.plot( da[-1000:].loc[:, :, 'leadrtn_adj_30min'].values.reshape( -1 ),
                 da[-1000:].loc[:, :, 'lagrtn_adj_30min'].values.reshape( -1 ), '.', alpha=0.3 )
        ax.set_xlabel( 'leading rtn' )
        ax.set_ylabel( 'lag rtn' )
        xcorr = (pd.Series( da.loc[:, :, 'leadrtn_adj_30min'].values.reshape( -1 ) )
                 .corr( pd.Series( da.loc[:, :, 'lagrtn_adj_30min'].values.reshape( -1 ) ) ))
        ax.set_title( f'lead vs lag (no shift) {100*xcorr:0.2f}%' )
        ax = subplot( 2, 2, 2 )
        dal = da.lag( 1 )
        ax.plot( dal[-1000:].loc[:, :, 'leadrtn_adj_30min'].values.reshape( -1 ),
                 da[-1000:].loc[:, :, 'lagrtn_adj_30min'].values.reshape( -1 ), '.', alpha=0.3 )
        ax.set_xlabel( 'leading rtn' )
        ax.set_ylabel( 'lag rtn' )
        xcorr = (pd.Series( dal.loc[:, :, 'leadrtn_adj_30min'].values.reshape( -1 ) )
                 .corr( pd.Series( da.loc[:, :, 'lagrtn_adj_30min'].values.reshape( -1 ) ) ))
        ax.set_title( f'lead vs lag (no shift) {100*xcorr:0.2f}%' )
        for i, sym in enumerate( ['SPY', 'GLD'] ):
            ax = subplot( 2, 2, 3 + i )
            sdf = da.loc[:, sym, ['lagrtn_adj_1day', 'leadrtn_adj_1day']].to_dataframe_mx()
            ax.plot( sdf['2018']['lagrtn_adj_1day'].rolling( 100 ).mean(), '-r', label='lagging' )
            ax.plot( sdf['2018']['leadrtn_adj_1day'].rolling( 100 ).mean(), '--b', alpha=0.8, label='leading' )
            ax.legend()
            ax.set_label( sym )
        fig.suptitle( 'leading vs lag returns ', fontsize=15 )
        figs.append( fig )
    
    with logger.timer( 'plotting returns results - 3' ):
        nc = da.loc[:, :, 'leadrtn_adj_30min'].abs().max( dim='time' )
        fig = figure( figsize=(10, 10) )
        for i, sym in enumerate( nc[nc > 1].symbol.values[:16] ):
            ax = subplot( 4, 4, i + 1 )
            sdf = da.loc[:, sym, ['lagrtn_adj_30min', 'leadrtn_adj_30min', 'close_adj_30min']].to_dataframe_mx()
            ax.plot( (1 + sdf['lagrtn_adj_30min']).cumprod() - 1, '.r', alpha=0.1 )
            ax.set_ylabel( 'cum rtn', color='red' )
            ax.twinx().plot( sdf['close_adj_30min'], '-k', label='close_adj' )
            ax.twinx().set_ylabel( 'open', color='black' )
            ax.set_title( sym )
        fig.suptitle( 'Symbols with bad returns', fontsize=15 )
        figs.append( fig )
    
    with logger.timer( 'plotting cross sectional statistics' ):
        features = ['lagrtn_adj_30min', 'leadrtn_adj_30min', 'weight', 'avg_dollar_volume']
        features.append( mx.Features( '/lagrtn_adj_30min.*/' ).find( ems.output_features )[0] )
        features.append( beta_computer.output_features[0] )
        fig = figure( figsize=(15, len( features ) * 2) )
        for i, ft in enumerate( features ):
            ax = subplot( len( features ) // 3 + 1, 3, i + 1 )
            wmean = da.loc[:, :, ft].mean( dim='symbol' ).to_series()
            wsd = da.loc[:, :, ft].std( dim='symbol' ).to_series()
            wmax = da.loc[:, :, ft].max( dim='symbol' ).to_series()
            wmin = da.loc[:, :, ft].min( dim='symbol' ).to_series()
            ax.fill_between( wmean.index, wmin, wmax, color='blue', alpha=0.5 )
            ax.fill_between( wmean.index, wmean - wsd, wmean + wsd, color='red', alpha=0.7 )
            ax.plot( wmean, '-k', alpha=1, linewidth=2 )
            ax.set_title( ft, fontsize=15 )
        fig.suptitle( 'Cross sectional stats (mean, 1sd, min, max)', fontsize=15 )
        figs.append( fig )
    
    with logger.timer( 'plotting betas' ):
        fig = figure( figsize=(15, 15) )
        for i, symbol in enumerate( ['SPY', 'AGG', 'GLD', 'XLF', 'BND', 'FDN'] ):
            ax = subplot( 3, 2, i + 1 )
            xy = ((da.loc[:, symbol, 'lagrtn_adj_30min'] * da.loc[:, symbol, 'mkt_lagrtn_30min'])
                  .to_series()
                  .ewm( halflife=beta_computer.halflifes[0], min_periods=1 )
                  .mean())
            xx = ((da.loc[:, symbol, 'mkt_lagrtn_30min'] ** 2)
                  .to_series()
                  .ewm( halflife=beta_computer.halflifes[0], min_periods=1 )
                  .mean())
            ax.plot( xy / (xx + beta_computer.eps), '-b', label='computed', linewidth=2 )
            ax.plot( beta_da.loc[:, symbol, beta_computer.output_features[0]].to_series(),
                     '--k', alpha=0.5, label=beta_computer.output_features[0] )
            ax.plot( beta_da.loc[:, symbol, beta_computer.output_features[1]].to_series(),
                     '--r', alpha=0.5, label=beta_computer.output_features[1] )
            ax.plot( beta_da.loc[:, symbol, beta_computer.output_features[2]].to_series(),
                     '--g', alpha=0.5, label=beta_computer.output_features[2] )
            ax.set_title( symbol )
        fig.suptitle( 'beta computer vs manually computed beta ( blue)', fontsize=15 )
        figs.append( fig )
    
    with logger.timer( 'plot results of residualization' ):
        
        def _roll_corr( X, Y, win=5 * 13 ):
            XY = (X * Y).mean( axis=1 )
            Xs = (X.sq()).mean( axis=1 ).sqrt()
            Ys = (Y.sq()).mean( axis=1 ).sqrt()
            XYc = 100 * (XY / (Xs + 1e-13) / (Ys + 1e-13))
            # apply a smoothing filter to the correlations
            return XYc.to_series().rolling( win, min_periods=1 ).mean(), XYc.mean().item()
        
        fig, axs = subplots( len( residualizer.output_features ), len( residualizer.features ), figsize=(20, 20) )
        for i, (yorig, yresid) in enumerate( zip( residualizer.response.cols, residualizer.output_features ) ):
            for j, x in enumerate( residualizer.features ):
                ax = axs[i][j]
                orc, oc = _roll_corr( X=da.loc[:, :, x], Y=da.loc[:, :, yorig] )
                rrc, rc = _roll_corr( X=da.loc[:, :, x], Y=da.loc[:, :, yresid] )
                ax.plot( orc.rolling( 13 * 20 ).mean(), '-r', alpha=0.8, label=f'raw {oc}' )
                ax.plot( rrc.rolling( 13 * 20 ).mean(), '--b', alpha=0.8, label=f'resid {rc}' )
                ax.set_xlabel( yorig, fontsize=12 )
                ax.set_title( x, fontsize=12 )
                ax.legend()
                ax.set_ylim( [-10, 10] )
        fig.suptitle( 'correlations before and after residualization', fontsize=15 )
        figs.append( fig )
    
    with logger.timer( ' verifing the results of the residualizer' ):
        factor_returns_computer = (
            mx.returns( sid='symbol' )
                .factor_returns( response=residualizer.response.cols + residualizer.output_features, weight='weight',
                                 updater=updater )
                .set( verbose=True )
        )
        fr_da = factor_returns_computer( da.fillna( 0 ), features=residualizer.features )
        
        fig, axs = subplots( len( residualizer.output_features ), len( residualizer.features ), figsize=(20, 20) )
        for i, (yorig, yresid) in enumerate( zip( residualizer.response.cols, residualizer.output_features ) ):
            for j, x in enumerate( residualizer.features ):
                ax = axs[i][j]
                orig_betas = fr_da.loc[:, x, yorig].to_series()
                resid_betas = fr_da.loc[:, x, yresid].to_series()
                oc = orig_betas.mean()
                rc = resid_betas.mean()
                ax.plot( orig_betas.rolling( 13 * 20 ).mean(), '-r', alpha=0.8, label=f'raw {oc}' )
                ax.plot( resid_betas.rolling( 13 * 20 ).mean(), '--b', alpha=0.8, label=f'resid {rc}' )
                ax.set_ylabel( yorig, fontsize=12 )
                ax.set_title( x, fontsize=12 )
                ax.legend()
        fig.suptitle( 'betas before and after residualization', fontsize=15 )
        figs.append( fig )
        
        fig = figure( figsize=(20, 20) )
        for i, sym in enumerate( ['SPY', 'AGG', 'GLD', 'XLF', 'BND', 'FDN'] ):
            ax = subplot( 3, 2, i + 1 )
            ax.plot( (1 + da.loc[:, sym, lagrtn].to_series()).cumprod() - 1, '-b', label='raw' )
            ax.plot( (1 + da.loc[:, sym, f'{lagrtn}_resid'].to_series()).cumprod() - 1, '-r', label='resid' )
            ax.set_title( sym )
            ax.legend()
        fig.suptitle( 'Raw and resid returns for some instruments', fontsize=15 )
        figs.append( fig )
    
    da = da.drop_mx( ['open_adj_30min', 'close_adj_30min', 'high_adj_30min', 'low_adj_30min', 'volume_adj_30min',
                      'dollar_volume_adj_30min', 'market', 'after_market', 'pre_market'] )
    return da, [mx.fig2format( fig ) for fig in figs]

