'''
load symbol data - and compute some interestign statistics

such as

* open - close
* high - low
* trailing min , trailing max price
* min / max price over a period
* time from previous high, low
* sign of above
* positive sign volatility
* negative sign volatiity
* +ve/-ve volume
* diff with min, max price


@date 5/18/2020

Goes with
http://localhost:18888/notebooks/research/etf_technical/etf_technical_2020.05.17%20(talib).ipynb#


'''
from pylab import *
from madmax.api import *


@mx.operatorize( consumes_features=False, produces_features=False, produces_meta=False, memoize=True )
def compute_returns_and_betas( da, start, end, dollar_volume_window, price_windows, beta_halflifes, resid_halflife ):
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
    
    with logger.timer( " measure universe coverage and intervals per day" ):
        valid_count = da.loc[:, :, 'valid_30min'].sum( dim='symbol' ).to_series()
        ints_every_day = da.time.to_dataframe().assign( date=lambda df: df.index.date ).groupby( 'date' ).count()
        assert all( ints_every_day == 13 ), 'not 13 intervals every day - fix it'
    
    price_timedeltas = [pd.Timedelta( win * 30, "m" ) for win in price_windows]
    with logger.timer( 'computing VWAPS' ):
        da = da.assign_features( mid_adj_30min=da.loc[:, :, ['open_adj_30min', 'close_adj_30min']].mean( axis=-1 ) )
        # compute VWAP prices
        price_sma = (mx.transforms.rolling( sid='symbol', windows=price_windows )
                     .mean( weight='dollar_volume_unadj_30min' ))
        da = price_sma( da, features=['mid_adj_30min'] )
        da = da.rename_coords( features={ of: f'mid_adj_{td}.vwap'
                                          for of, td in zip( price_sma.output_features, price_timedeltas ) } )
    
    with logger.timer( 'adding price features' ):
        da = da.assign_features(
            true_range_30min=((da.loc[:, :, 'high_adj_30min'] - da.loc[:, :, 'low_adj_30min'])
                              / (da.loc[:, :, 'close_adj_30min'] + 1e-12)).clip( -3, 3 ),
            direction_30min=(da.loc[:, :, f'close_adj_30min'] - da.loc[:, :, f'open_adj_30min']).sign(),
            delta_rtn_30min=((da.loc[:, :, 'close_unadj_30min'] - da.loc[:, :, 'open_unadj_30min'])
                             / (da.loc[:, :, 'open_unadj_30min'] + 1e-12)).clip( -3, 3 ),
            log_open_unadj_30min=da.loc[:, :, 'open_unadj_30min'].log(),
            log_close_unadj_30min=da.loc[:, :, 'close_unadj_30min'].log(),
        )
        pda = (da.loc[:, :, ['open_adj_30min', 'high_adj_30min', 'low_adj_30min', 'close_adj_30min', ]]
               .ffill( dim='time', limit=13 * 3 )
               .fillna( 0 ))
        # add high, low, open and close over other windows - along with distance since last high/low
        for win, td in zip( price_windows, price_timedeltas ):
            da = da.assign_features( **{
                f'open_adj_{td}': pda.lag( win ).loc[:, :, 'open_adj_30min'],
                f'high_adj_{td}': pda.loc[:, :, 'high_adj_30min'].rolling( time=win ).max(),
                f'low_adj_{td}': pda.loc[:, :, 'low_adj_30min'].rolling( time=win ).min(),
                # time to previous (as a fraction of total window length) high or low in the window
                # da.rolling.construct :  0-th index is the most distant and -1 is the most recent.
                f'ttp_high_adj_{td}': 1 - (pda.loc[:, :, 'high_adj_30min']
                                           .rolling( time=win ).construct( '_' ).argmax( axis=-1 ) / (win - 1)),
                f'ttp_low_adj_{td}': 1 - (pda.loc[:, :, 'low_adj_30min']
                                          .rolling( time=win ).construct( '_' ).argmin( axis=-1 ) / (win - 1)),
            } )
            da = da.assign_features( **{
                f'true_range_{td}': ((da.loc[:, :, f'high_adj_{td}'] - da.loc[:, :, f'low_adj_{td}'])
                                     / (da.loc[:, :, f'mid_adj_{td}.vwap'] + 1e-12)).clip( -3, 3 ),
                f'direction_{td}': (da.loc[:, :, f'close_adj_30min'] - da.loc[:, :, f'open_adj_{td}']).sign(),
                f'delta_rtn_{td}': ((da.loc[:, :, 'close_adj_30min'] - da.loc[:, :, f'open_adj_{td}'])
                                    / (da.loc[:, :, f'open_adj_{td}'] + 1e-12)).clip( -3, 3 ),
            } )
    
    with logger.timer( 'compute returns' ):
        for sfx, periods in logger.pbar( [('30min', 1), ('1hr', 2), ('1day', 13)] ):
            # fjnote - a tighter clamp on max returns (30% in 30mins)
            lagrtn_computer = mx.returns( sid='symbol' ).lagging_da( periods=periods, max_gap=13 * 3,
                                                                     max_rtn=periods * 0.2 )
            leadrtn_computer = mx.returns( sid='symbol' ).leading_da( periods=periods, max_gap=13 * 3,
                                                                      max_rtn=periods * 0.2 )
            # @5/20/2020 Modified this code to use open for lagging returns and close for leading returns
            da = lagrtn_computer( da, features='open_adj_30min' )
            da = leadrtn_computer( da, features='close_adj_30min' )
            da = da.rename_coords( features={ lagrtn_computer.output_features[0]: f'lagrtn_adj_{sfx}',
                                              leadrtn_computer.output_features[0]: f'leadrtn_adj_{sfx}' } )
    
    with logger.timer( 'adding volume, weight and vol features' ):
        da = da.assign_features(
            log10_dollar_volume_30min=da.loc[:, :, 'dollar_volume_unadj_30min'].log10(),
            mkt_lagrtn_30min=da.loc[:, 'SPY', 'lagrtn_adj_30min'],
            mkt_leadrtn_30min=da.loc[:, 'SPY', 'leadrtn_adj_30min'],
            mkt_leadrtn_1day=da.loc[:, 'SPY', 'leadrtn_adj_1day'],
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
    gc.collect()
    
    with logger.timer( 'computing market betas and market return ' ):
        beta_computer = mx.returns( sid='symbol' ).factor_loadings( factor_rtn='mkt_lagrtn_30min',
                                                                    halflifes=beta_halflifes )
        beta_da = beta_computer( da, features='lagrtn_adj_30min' )
        da = da.assign_features( betas=beta_da )
    
    with logger.timer( 'computing residual returns' ):
        updater = mx.online.LinearUpdater( loss='l2', ridge=1e-5, algo='direct', hl_1=resid_halflife,
                                           hl_2=resid_halflife )
        residualizer = (mx.returns( sid='symbol' )
                        .residualize( response=['lagrtn_adj_30min', 'log10_dollar_volume_30min'],
                                      weight='weight', updater=updater )
                        .set( verbose=True ))
        da_rsd = residualizer( da.fillna( 0 ), features=['valid_30min'] + beta_computer.output_features )
        da = xa.concat( [da, da_rsd.loc[:, :, residualizer.output_features]], dim='features' )
    da = da.drop_mx( ['market_30min'] )
    return da


@mx.operatorize( consumes_features=False, produces_features=False, produces_meta=False, memoize=True )
def add_relative_price_and_volumes( da, emas ):
    '''add some price and volume features'''
    logger = mx.Logger( 'add_relative_price_and_volumes', verbose=True )
    # using backward adjusted returns is ok for ratios since the adjustment factors cancel out
    with logger.timer( 'add relative prices' ):
        price_emas = mx.transforms.rolling( sid='symbol', windows=emas ).mean( weight='dollar_volume_unadj_30min' )
        dao = price_emas( da, features='mid_adj_30min' )
        for ema, op in zip( emas, price_emas.output_features ):
            da = da.assign_features( **{ f'rel_price_{ema}': da.loc[:, :, 'mid_adj_30min'] / dao.loc[:, :, op] } )
    with logger.timer( 'add relative volumes' ):
        volume_emas = mx.transforms.rolling( sid='symbol', windows=emas ).mean()
        dao = volume_emas( da, features='dollar_volume_unadj_30min' )
        for ema, op in zip( emas, volume_emas.output_features ):
            da = da.assign_features( **{ f'rel_dollar_volume_{ema}':
                                             da.loc[:, :, 'dollar_volume_unadj_30min'] / dao.loc[:, :, op] } )
    return da


@mx.operatorize( consumes_features=False, produces_features=False, produces_meta=False, memoize=True )
def add_volatilities( da, emas ):
    '''add some price and volume features'''
    logger = mx.Logger( 'add_volatilities', verbose=True )
    with logger.timer( 'adding volatility features' ):
        rtn_vol = mx.transforms.rolling( sid='symbol', windows=emas ).sd( weight='weight' )
        da = rtn_vol( da, features=['lagrtn_adj_30min', 'lagrtn_adj_30min_resid', 'mkt_lagrtn_30min'] )
    with logger.timer( 'adding returns zscores' ):
        rtn_z = mx.transforms.rolling( sid='symbol', windows=emas ).zscore( weight='weight' )
        da = rtn_z( da, features=['lagrtn_adj_30min', 'lagrtn_adj_30min_resid'] )
    with logger.timer( 'adding calendar features' ):
        cf_df = mx.calendar_features( da.time.to_dataframe() ).drop_mx( 'time' )
        cf_da = cf_df.to_dataarray_mx()
        da = da.assign_features( cal_features=cf_da.rename( { cf_da.dims[-1]: da.dims[-1] } ) )
    return da


@mx.operatorize( consumes_features=True, produces_features=True, produces_meta=False, memoize=True )
def add_dijoint_returns( da, features, lags=(1, 2, 3) ):
    '''add dijsoint 30min leading returns'''
    output_features = []
    for feature in mx.make_iterable( features ):
        for lag in make_iterable( lags ):
            of = f'{feature}_disjoint.{lag}'
            da = da.assign_features( **{ of: da.loc[:, :, feature].lag( by=-lag ) } )
            output_features.append( of )
    return da, output_features



@mx.operatorize( consumes_features=True, produces_features=False, memoize=False )
def build_daily_data( data, features, timeofday ):
    tix = pd.to_datetime( data.time.values )
    # add 1day disjoint return shifted by 30mins and market return
    data = data.assign_features(
        leadrtn_adj_1day_disjoint_30min= data.loc[:, :, 'leadrtn_adj_1day'].lag( by=-1 ),
        mkt_lagrtn_1day=data.loc[:, 'SPY', 'lagrtn_adj_1day'],
    )
    # apply 1 day averages to 30min features
    ema = mx.transforms.rolling( windows=13, sid='symbol' ).mean( )
    data = ema( data, features=features )
    return data[tix.time == timeofday]