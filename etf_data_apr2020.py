'''
Code to process the ETF data downloaded on Apr 30 2020 (2020.04.30)

>>> http://www.kibot.com/Support.aspx#data_format
Market sessions
Intraday data records have a time stamp indicating the time when the bar opened.
For example, a time stamp of 10:00 AM is for a period between 10:00 AM and 10:01 AM.
All records with a time stamp between 9:30 AM and 3:59 PM represent the regular US trading session.
Our stock and etf data includes pre-market (8:00-9:30 a.m. ET), regular (9:30 a.m. -4:00 p.m. ET.)
and after market (4:00-6:30 p.m. ET.) sessions. Data records for SPY and some other liquid ETFs and
stocks usually start at 4 a.m and ends at 8 p.m.

'''
from madmax.api import *
from pylab import *
import pickle, glob
from research.yahoo import get_yahoo_data

KIBOT_DIR = os.path.join( HOMEDIR, 'data/kibot/' )
DATA_30MIN_DIR = os.path.join( KIBOT_DIR, 'all_symbols_2020.04.30/30min/' )
DATA_DAILY_DIR = os.path.join( KIBOT_DIR, 'all_symbols_2020.04.30/daily/' )

# ETF metadata lists
bond_etfs = pd.read_excel( HOMEDIR + '/data/all-etfs.xlsx', sheet_name='bond-etfs' )
equity_etfs = pd.read_excel( HOMEDIR + '/data/all-etfs.xlsx', sheet_name='equity' )
vol_etfs = pd.read_excel( HOMEDIR + '/data/all-etfs.xlsx', sheet_name='vol' )

etf_meta = pd.concat( [
    bond_etfs[['Symbol', 'ETP Name', 'Leveraged / Inverse', 'Asset Class']],
    equity_etfs[['Symbol', 'ETP Name', 'Leveraged / Inverse', 'Asset Class']],
    vol_etfs[['Symbol', 'ETP Name', 'Leveraged / Inverse', 'Asset Class']],
], axis=0 )

all_etfs = (pd.read_csv( KIBOT_DIR + 'all_etfs.csv' )
            .assign( Symbol=lambda df: df['Symbol'].map( lambda x: x.replace( '\xa0', '' ).strip() ) )
            .merge( etf_meta, on='Symbol', how='left' ))

all_etf_symbols = [symbol.replace( '\xa0', '' ).strip() for symbol in sorted( all_etfs.Symbol.values )]


@mx.memoize
def compute_etf_stats( all_etf_symbols=all_etf_symbols ):
    '''preference on stats after 20160101'''
    logger = mx.Logger( 'compute_etf_stats' )
    all_stats = []
    for symbol in logger.pbar( all_etf_symbols ):
        try:
            df = pd.read_csv( DATA_DAILY_DIR + f'/{symbol}.csv.bz2' )
            df['date'] = pd.to_datetime( df['date'] )
            df.index = df['date']
            stat = dict(
                symbol=symbol, start=df.date.min(), end=df['date'].max(),
                daily_dollar_volume_adj=(df['volume_adj'] * df['open_adj'])['20160101':].median(),
                daily_dollar_volume_unadj=(df['volume_unadj'] * df['open_unadj'])['20160101':].median(),
                n=len( df ), vendor='kibot'
            )
        except FileNotFoundError:
            logger.warn( f'kibot data for {symbol} not found - pulling from yahoo' )
            try:
                df = get_yahoo_data( symbol, start='19980101', end='20201201' )
                stat = dict(
                    symbol=symbol, start=df.index.min(), end=df.index.max(),
                    daily_dollar_volume_adj=(df['volume'] * df['open'])['20160101':].median(),
                    vendor='yahoo', n=len( df )
                )
            except:
                logger.warn( f'yahoo data for {symbol} not found - SKIPPING' )
                stat = dict( symbol=symbol, vendor=None )
        all_stats.append( stat )
    return pd.DataFrame( all_stats )


@mx.memoize
def get_data_for_symbol( symbol, freq='30min', verbose=False ):
    '''
    note the kibot data is time-stamped to the start of the interval - so technically the high, low, close and
    volume are not visible to it.
    
    So we can treat the kibot data as off by 30mins systematically with:
        ohlcv bars @ 0930 -> visible at 1000
        ohlcv bars @ 1530 -> visible at 4pm
        
    Therefore,
        We'll shift it by 30mins to make sure that this problem does not exist and now the bar is observed at the
        reported time. So now market bars are from 1000hrs to 1600hrs
    
    '''
    logger = mx.Logger( f'get_data_for_{symbol} @ {freq}', verbose=verbose )
    columns = ['open', 'high', 'low', 'close', 'volume']
    with logger.timer( 'building 30min data' ):
        df_30min = pd.read_csv( DATA_30MIN_DIR + f'{symbol}.csv.bz2' )
        df_30min['time'] = pd.to_datetime( df_30min['time'] )
        df_30min['date'] = pd.to_datetime( df_30min['date'] )
        assert (
                all( pd.to_datetime( df_30min['time'].dt.date ) == df_30min['date'] )
                and df_30min['time'].is_monotonic_increasing
        ), 'fuckup in time stamaps'
        # do the time stamp adjustment for the 30min data
        df_30min['time'] = df_30min['time'] + pd.Timedelta( 30, 'm' )
        df_30min = df_30min.set_index( 'time' )
        df_30min = df_30min.rename(
            columns={ f'{col}_{adj}': f'{col}_{adj}_30min' for col in columns for adj in ['adj', 'unadj'] }
        )
        df_30min['dollar_volume_unadj_30min'] = df_30min['open_unadj_30min'] * df_30min['volume_unadj_30min']
        df_30min['dollar_volume_adj_30min'] = df_30min['open_adj_30min'] * df_30min['volume_adj_30min']
        df_30min['pre_market'] = df_30min.time_index.time <= dt.time( 9, 30 )
        df_30min['market'] = ((df_30min.time_index.time > dt.time( 9, 30 ))
                              & (df_30min.time_index.time <= dt.time( 16, 0 )))
        df_30min['after_market'] = df_30min.time_index.time > dt.time( 16, 0 )
        df_30min['valid'] = True
        ols = sm.OLS( df_30min[df_30min['market']]['dollar_volume_unadj_30min'],
                      df_30min[df_30min['market']]['dollar_volume_adj_30min'], missing='drop' ).fit()
        if not np.allclose( ols.params[0], 1, rtol=1e-3 ):
            logger.warn( 'messup in computing dollar volume, beta=', ols.params[0] )
    with logger.timer( 'building pre and post market data ' ):
        premkt = { }
        postmkt = { }
        mkt = { }
        for _mkt, _df in [(mkt, df_30min[df_30min['market']]), (premkt, df_30min[df_30min['pre_market']]),
                          (postmkt, df_30min[df_30min['after_market']])]:
            for adj in ['adj', 'unadj']:
                _mkt[f'open_{adj}_30min'] = _df.groupby( 'date' )[f'open_{adj}_30min'].first()
                _mkt[f'close_{adj}_30min'] = _df.groupby( 'date' )[f'close_{adj}_30min'].last()
                _mkt[f'high_{adj}_30min'] = _df.groupby( 'date' )[f'high_{adj}_30min'].max()
                _mkt[f'low_{adj}_30min'] = _df.groupby( 'date' )[f'low_{adj}_30min'].min()
                _mkt[f'volume_{adj}_30min'] = _df.groupby( 'date' )[f'volume_{adj}_30min'].sum()
                _mkt[f'dollar_volume_{adj}_30min'] = _df.groupby( 'date' )[f'dollar_volume_{adj}_30min'].sum()
        df_premkt = pd.DataFrame( premkt ).rename( columns=lambda x: x.replace( '30min', 'premkt' ) )
        df_postmkt = pd.DataFrame( postmkt ).rename( columns=lambda x: x.replace( '30min', 'postmkt' ) ).shift( 1 )
        df_mkt = pd.DataFrame( mkt ).rename( columns=lambda x: x.replace( '30min', 'mkt' ) )
    with logger.timer( '# pull the daily data' ):
        df_daily = pd.read_csv( DATA_DAILY_DIR + f'{symbol}.csv.bz2' )
        df_daily['date'] = pd.to_datetime( df_daily['date'] )
        df_daily['time'] = df_daily['date'] + pd.Timedelta( 16, 'H' )
        df_daily = df_daily.set_index( 'date' )
        df_daily = df_daily.rename(
            columns={ f'{col}_{adj}': f'{col}_{adj}_day' for col in columns for adj in ['adj', 'unadj'] } )
    # compare 30min with the daily numbers
    with logger.timer( 'comparing 30min with daily data' ):
        jdf = df_mkt.merge( df_daily, left_index=True, right_index=True )
        comp_stats = []
        for c in columns:
            for adj in ['adj', 'unadj']:
                ols = sm.OLS( jdf[f'{c}_{adj}_mkt'], jdf[f'{c}_{adj}_day'], missing='drop' ).fit()
                comp_stats.append(
                    dict( symbol=symbol, adj=adj, feature=c, beta=ols.params[0], r=np.sqrt( ols.rsquared ) ) )
        comp_stats = pd.DataFrame( comp_stats )
    return df_30min if freq == '30min' else df_daily, comp_stats


@mx.operatorize( consumes_features=False, produces_features=False, produces_meta=True, memoize=True )
def Concat30minData( data, symbols ):
    '''concat all the 30min data together and return a data'''
    logger = mx.Logger( 'get_data_for_symbol', verbose=True )
    das = []
    metas = []
    for i, symbol in logger.pbar( enumerate( symbols ) ):
        df, meta = get_data_for_symbol( symbol, freq='30min' )
        df = df.set_index( 'symbol', append=True )
        das.append( df.drop_mx( ['date', 'timeofday'] ).to_dataarray_mx() )
        metas.append( meta )
    with logger.timer( 'concating ' ):
        da = xa.concat( das, dim='symbol' )
    da.name = '30min_data'
    meta = pd.concat( metas, axis=0 ).set_index( ['symbol', 'adj', 'feature'] ).unstack( level=[1, 2] )
    meta.columns = meta.columns.swaplevel( 0, 2 )
    return da, meta.sort_index( axis=1 )


def compare_adj_vs_unadj( symbol ):
    '''self explanatory'''
    adf = pd.read_csv( DATA_DAILY_DIR + f'{symbol}.csv.bz2' )
    adf = adf.set_index( pd.to_datetime( adf.date ) )
    
    ddf = pd.read_csv( DATA_30MIN_DIR + f'{symbol}.csv.bz2' )
    ddf = ddf.set_index( pd.to_datetime( ddf.time ) )
    
    subplot( 231 )
    plot( ddf.open_adj, '-r' )
    plot( ddf.open_unadj, '-b' )
    gca().set_title( '30min open' )
    
    subplot( 232 )
    plot( ddf.volume_adj.log10().rolling( 13 * 100 ).mean(), '-r' )
    plot( ddf.volume_unadj.log10().rolling( 13 * 100 ).mean(), '-b' )
    gca().set_title( '30min volume' )
    
    subplot( 233 )
    plot( (ddf.volume_adj * ddf.open_adj).log10().rolling( 13 * 100 ).mean(), '-r', label='adj' )
    plot( (ddf.volume_unadj * ddf.open_unadj).log10().rolling( 13 * 100 ).mean(), '-b', label='unadj' )
    gca().set_title( '30min dollar volume' )
    
    subplot( 234 )
    plot( adf.open_adj, '-r' )
    plot( adf.open_unadj, '-b' )
    gca().set_title( 'daily open' )
    
    subplot( 235 )
    plot( adf.volume_adj.log10().rolling( 100 ).mean(), '-r', )
    plot( adf.volume_unadj.log10().rolling( 100 ).mean(), '-b' )
    gca().set_title( 'daily volume' )
    
    subplot( 236 )
    plot( (ddf.volume_adj * ddf.open_adj).log10().rolling( 13 * 100 ).mean(), '-r', label='adj' )
    plot( (ddf.volume_unadj * ddf.open_unadj).log10().rolling( 13 * 100 ).mean(), '-b', label='unadj' )
    gca().set_title( 'daily dollar volume' )

@mx.memoize
def get_data_for_symbol_with_afacs( symbol, mkt_only=True, verbose=False, start='20030101' ):
    '''get the data for one symbol along with adjustment factors'''
    
    logger = mx.Logger( f'get_data_for_symbol_with_afacs_{symbol} ', verbose=verbose )
    
    # load the 30min and daily data with backward adjustments
    columns = ['open', 'high', 'low', 'close', 'volume']
    with logger.timer( 'building 30min data' ):
        df_30min = (
            pd.read_csv( DATA_30MIN_DIR + f'{symbol}.csv.bz2' )
            .assign(
                time=lambda df: pd.to_datetime( df['time'] ),
                date=lambda df: pd.to_datetime( df['date'] ),
                mid_unadj=lambda df: df[['open_unadj', 'close_unadj']].mean( axis=1 ),
                dollar_volume_unadj=lambda df: df['mid_unadj'] * df['volume_unadj'],
            )
        )
        assert (
                all( pd.to_datetime( df_30min['time'].dt.date ) == df_30min['date'] )
                and df_30min['time'].is_monotonic_increasing
        ), 'fuckup in time stamaps'
        # do the time stamp adjustment for the 30min data
        df_30min['time'] = df_30min['time'] + pd.Timedelta( 30, 'm' )
        df_30min['pre_market'] = df_30min.time.dt.time <= dt.time( 9, 30 )
        df_30min['market'] = ((df_30min.time.dt.time > dt.time( 9, 30 ))
                              & (df_30min.time.dt.time <= dt.time( 16, 0 )))
        df_30min['after_market'] = df_30min.time.dt.time > dt.time( 16, 0 )
        df_30min = df_30min[df_30min.time > pd.to_datetime( start )]

    if mkt_only:
        df_30min = df_30min[df_30min['market']].drop( columns=['pre_market', 'after_market'] )
        
    with logger.timer( '# pull the daily data' ):
        df_daily = pd.read_csv(
            DATA_DAILY_DIR + f'{symbol}.csv.bz2'
        ).assign(
            date=lambda df: pd.to_datetime( df['date'] ),
            time=lambda df: df['date'] + pd.Timedelta( 16, 'H' ),
            mid_unadj=lambda df: df[['open_unadj', 'close_unadj']].mean( axis=1 ),
            dollar_volume_unadj=lambda df: df['mid_unadj'] * df['volume_unadj'],
        )
        df_daily = df_daily[df_daily.time > pd.to_datetime( start )]

    missing = ((df_daily.isnull().sum( axis=1 ) > 0).sum() / len( df_daily ) * 100,
               (df_30min.isnull().sum( axis=1 ) > 0).sum() / len( df_30min ) * 100)
    if any( [m > 10 for m in missing] ):
        logger.warn( f'missing adjustment {missing}% data for {symbol} - skipping' )
        return None, None
    
    def compute_afacs( df ):
        ''' compute the forward and backward adjustment factor '''
        afac_df = df[[]].assign(
            open_bk=df['open_adj'] / df['open_unadj'],
            high_bk=df['high_adj'] / df['high_unadj'],
            low_bk=df['low_adj'] / df['low_unadj'],
            close_bk=df['close_adj'] / df['close_unadj'],
            # this one is a bit problematic
            # volume_bk=df['volume_unadj'] / df['volume_adj'],
        ).assign(
            bk_afac=lambda df: df.median( axis=1 )
        )
        # test the correctness of this data
        cov = afac_df.cov()
        covs = cov.unstack().values
        if not ( np.allclose( covs.max(), covs.min(), rtol=1e-2 )
                 or np.allclose( covs.max(), covs.min(), atol=1e-3 ) ):
            logger.warn( 'imprecise afacs for', symbol, cov )
            raise ValueError( f'failure in computing adjustment factors for {symbol}', cov )
        # the forward adjustment factor
        df['bw_afac'] = afac_df['bk_afac']
        df['fw_afac'] = afac_df['bk_afac'] / afac_df['bk_afac'].iloc[0]
        return df
    
    # compute forward adjusted prices and volumes
    df_30min = compute_afacs( df_30min )
    df_daily = compute_afacs( df_daily )
    
    # compare daily and 30min data
    _df = df_30min[df_30min['market']]
    _mkt = { }
    for adj in ['adj', 'unadj']:
        _mkt[f'open_{adj}'] = _df.groupby( 'date' )[f'open_{adj}'].first()
        _mkt[f'close_{adj}'] = _df.groupby( 'date' )[f'close_{adj}'].last()
        _mkt[f'high_{adj}'] = _df.groupby( 'date' )[f'high_{adj}'].max()
        _mkt[f'low_{adj}'] = _df.groupby( 'date' )[f'low_{adj}'].min()
        _mkt[f'volume_{adj}'] = _df.groupby( 'date' )[f'volume_{adj}'].sum()
    for afac in ['fw_afac', 'bw_afac']:
        _mkt[afac] = _df.groupby( 'date' )[afac].mean()
    df_mkt = pd.DataFrame( _mkt ).reset_index()
    
    # test 30min and daily data
    jdf = df_mkt.merge( df_daily, how='inner', on='date', suffixes=('_mkt', '_day') )
    test_30_vs_day = { }
    for col in columns:
        for adj in ['adj', 'unadj']:
            ols = sm.OLS( jdf[f'{col}_{adj}_mkt'], jdf[f'{col}_{adj}_day'], missing='drop' ).fit()
            test_30_vs_day[f'{col}_{adj}'] = ols.params[0]
    for afac in ['fw_afac', 'bw_afac']:
        ols = sm.OLS( jdf[f'{afac}_mkt'], jdf[f'{afac}_day'], missing='drop' ).fit()
        test_30_vs_day[afac] = ols.params[0]
    test_30_vs_day = pd.DataFrame( test_30_vs_day, index=[symbol] )
    if any( test_30_vs_day.drop_mx( ['/volume.*/', 'fw_afac'] ).min() < 0.99 ):
        logger.warn( test_30_vs_day )
        raise ValueError( f'mismatch between daily and 30min data for {symbol}', test_30_vs_day )
    
    def clean_up( df, freq ):
        return (df
                .drop_mx( ['volume_adj', 'timeofday', 'date'] )
                .set_index( ['time', 'symbol'] )
                .sort_index()
                .assign( valid=1.0 )
                .rename( columns=lambda x: f'{x}_{freq}' ))
    
    return clean_up( df_30min, '30min' ), clean_up( df_daily, 'day' )


@mx.operatorize( consumes_features=False, produces_features=False, produces_meta=True, memoize=True )
def BuildDataWithAfacs( data, symbols, mkt_only=True, start='20030101' ):
    '''concat all the 30min data together and return a dataarray. Return the daily data as a dataframe'''
    logger = mx.Logger( 'get_data_for_symbol', verbose=True )
    das = []
    metas = []
    for i, symbol in logger.pbar( enumerate( symbols ) ):
        try:
            df_30min, df_daily = get_data_for_symbol_with_afacs( symbol, mkt_only=mkt_only, start=start )
            if df_30min is None:
                logger.warn( f'could not get data for {symbol} - SKPPING' )
                continue
        except ValueError as e:
            logger.warn( f'FAILED TO not get data for {symbol} - SKPPING because of ', e )
            continue
        das.append( df_30min.to_dataarray_mx() )
        metas.append( df_daily )
    with logger.timer( 'concating ' ):
        da = xa.concat( das, dim='symbol' )
    da.name = '30min_data'
    meta = pd.concat( metas, axis=0 )
    return da, meta.sort_index()


######################################################################################################
# -- compare the new data with the  old kibot data from 2020---
######################################################################################################

from research.etf_technical import load_feb2020_data

@mx.memoize
def load_kibot_etf_data_from_feb2020( verbose=False ):
    '''this loads the old kibot data that was pulled by randhir in Feb 2020-ish'''
    das = []
    logger = mx.Logger( 'load_and_concat_data', verbose=verbose )
    with logger.timer( 'loading the data'):
        for symbol in logger.pbar( load_feb2020_data.get_saved_30min_data.all_symbols ):
            if symbol.startswith( 'INCOMPLETE' ):
                logger.warn( f'No data for {symbol} - SKIPPING' )
                continue
            logger.info( 'loading data for ', symbol )
            da, meta = load_feb2020_data.get_saved_30min_data( symbol=symbol )
            # add a mask to indicate when this feature was valid
            da = da.assign_features( valid=da.loc[:,:,'open_30min'].isfinite() )
            das.append( da )
        da = xa.concat( das, dim='symbol' )
    # add an intercept column
    da = da.assign_features( one=1 )
    return da


def compare_with_kibot_data_from_feb2020():
    '''--- compare the new data with the  old kibot data from 2020---    '''
    da_old = load_kibot_etf_data_from_feb2020()
    js = sorted(set(da.symbol.values).intersection( da_old.symbol.values ))
    jt = pd.to_datetime( sorted(set(da.time.values).intersection( da_old.time.values )) )
    
    dda = da.loc[jt,js,'volume_adj_30min']*da.loc[jt,js,'open_adj_30min']
    ddao = da_old.loc[jt,js,'volume_30min']*da_old.loc[jt,js,'open_30min']
    
    plot( ddao.values.reshape(-1), dda.values.reshape(-1), '.', alpha=0.3 )