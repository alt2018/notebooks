# load the kibot and pitrading data and compare against polygon
# this was using
#    the kibot data downloaded in Feb 2020 (by Randhir) for only top 200 ETFS and stocks (adjusted prices only)
#    polygon data from Aug 2019 (download by me)
#    pitrading data downloaded by randhir

# these datasets contain only adjusted prices which are problematic
#####################################################################################################

from madmax.api import *
import pickle, glob

# ETF metadata lists
bond_etfs = pd.read_excel( HOMEDIR + '/data/all-etfs.xlsx', sheet_name='bond-etfs' )
equity_etfs = pd.read_excel( HOMEDIR + '/data/all-etfs.xlsx', sheet_name='equity' )
vol_etfs = pd.read_excel( HOMEDIR + '/data/all-etfs.xlsx', sheet_name='vol' )

# -- all these files have been archived as tar.gz--
pidata_dir = os.path.join( HOMEDIR, 'data/pi/' )
pidata_files = OrderedDict(
    # [(f.replace( '.csv', '' ), os.path.join( pidata_dir, f )) for f in os.listdir( pidata_dir ) if
    #  re.fullmatch( '.*csv', f )]
)

kibot_day_dir = os.path.join( HOMEDIR, 'data/kibot/etf_top200_eod_adj' )
kibot_5min_dir = os.path.join( HOMEDIR, 'data/kibot/etf_top200_5min' )
kibot_day_files = OrderedDict(
    # [(f.replace( '\xa0', '' ).replace( '.csv', '' ), os.path.join( kibot_day_dir, f )) for f in
    #  os.listdir( kibot_day_dir ) if re.fullmatch( '.*csv', f )]
)
kibot_5min_files = OrderedDict(
    # [(f.replace( '\xa0', '' ).replace( '.csv', '' ), os.path.join( kibot_5min_dir, f )) for f in
    #  os.listdir( kibot_5min_dir ) if re.fullmatch( '.*csv', f )]
)
# assert len( set( kibot_5min_files.keys() ).difference( kibot_day_files.keys() ) ) == len(
#     set( kibot_day_files.keys() ).difference( kibot_5min_files.keys() ) ) == 0, 'ticker mismatch'



@mx.memoize
def etf_data( symbol, output='kibot_5min', verbose=True ):
    '''
    process one symbol from polygon, kibot and pitrading
    '''
    assert output in ['kibot_5min', 'kibot_day', 'pi'], 'bad output type'
    # get metadata for the symbol
    if symbol.startswith( 'INCOMPLETE' ):
        return None, { 'symbol': symbol }
    meta = { 'ETP Name': 'Not Found', 'Asset Class': 'Not Found' }
    for etf_list in [bond_etfs, equity_etfs, vol_etfs]:
        if symbol in etf_list.Symbol.values:
            meta = etf_list[etf_list.Symbol == symbol].iloc[0].to_dict()
            break
    meta = dict( symbol=symbol, name=meta['ETP Name'], asset_class=meta['Asset Class'] )
    if output == 'kibot_day':
        kdf_day = pd.read_csv(
            kibot_day_files[symbol], header=None,
            names=['date', 'open', 'high', 'low', 'close', 'volume']
        ).assign( vendor='kibot_day', **meta )
        kdf_day['time'] = pd.to_datetime( kdf_day.date, format='%m/%d/%Y' ) + pd.Timedelta( 16, 'H' )
        kdf_day['dollar_volume'] = kdf_day['volume'] * kdf_day['open']
        return kdf_day, meta
    if output == 'kibot_5min':
        kdf_5min = pd.read_csv(
            kibot_5min_files[symbol], header=None,
            names=['date', 'timedelta', 'open', 'high', 'low', 'close', 'volume']
        ).assign( vendor='kibot_5min', **meta )
        kdf_5min['dollar_volume'] = kdf_5min['volume'] * kdf_5min['open']
        kdf_5min['time'] = (pd.to_datetime( kdf_5min.date, format='%m/%d/%Y' )
                            + pd.to_timedelta( kdf_5min.timedelta + ':00' ))
        return kdf_5min, meta
    # pitrading data
    if output == 'pi' and symbol in pidata_files:
        pidf = pd.read_csv( pidata_files[symbol] ).assign( vendor='pi_1min', **meta )
        pidf.columns = [c.lower() for c in pidf.columns]
        pidf = pidf.rename( columns={ 'time': 'timedelta' } )
        pidf['time'] = (
                pd.to_datetime( pidf.date, format='%m/%d/%Y' )
                + pd.to_timedelta( pidf.timedelta.map( lambda x: '{:02}:{:02}:00'.format( x // 100, x % 100 ) ) )
        )
        pidf['dollar_volume'] = pidf['volume'] * pidf['open']
        return pidf, meta


# to run all the ETF symbols
# etf_ops = [etf_data( symbol=symbol, output='kibot_day' ).set( verbose=True, data=chain[-1] ) for symbol in pbar( sorted( kibot_day_files ) )]
# outs = ( [op( ) for op in pbar(etf_ops) ] )


########################################################################################
# --- compute returns at 5 min frequ-----
########################################################################################

@mx.memoize
def compute_returns( symbol, mkt_hours_only=True, plots=False, verbose=False ):
    '''
    
    Parameters
    ----------
    symbol
    mkt_hours_only
        if True - restrict results to market hours only
    plots
    verbose

    Returns
    -------
    
    
    Test
    ----
    ```
        df, da = eld.compute_returns( symbol='SPY' )
        
        with mx.timer( 'sanity checking returns' ):
            figure()
            subplot( 221 )
            plot( da.loc[:, :, 'open_5min'].values.reshape( -1 ),
                  da.loc[:, :, 'logrtn_lag_5min'].cumsum().exp().values.reshape( -1 ),
                  '.b', alpha=0.3 )
            gca().set_title( 'cum returns vs open' )
            
            subplot( 222 )
            plot( da.loc[:, :, 'logrtn_lag_5min'].values.reshape( -1 ),
                  da.loc[:, :, 'logrtn_lead_5min'].shift( time=1 ).values.reshape( -1 ), '.' )
            gca().set_title( 'leading vs lagg' )
            
            subplot( 223 )
            plot( da.loc[:, :, 'close_prev_day'].values.reshape( -1 ),
                  da.loc[:, :, 'close_day'].lag( '1D' ).values.reshape( -1 ), '.' )
            gca().set_title( 'curr vs prev day close' )
            
            subplot( 224 )
            plot( da.loc[:, :, 'close_prev_week'].values.reshape( -1 ),
                  da.loc[:, :, 'close_week'].values.reshape( -1 ), '.' )
            gca().set_title( 'curr vs prev week close' )
    ```
    '''

    INTERVALS_PER_HOUR = 12
    INTERVALS_PER_DAY = int( 6.5 * 12 )
    logger = mx.Logger( 'compute_returns', verbose=verbose )
    if symbol.startswith( 'INCOMPLETE' ):
        logger.warn( f'No data for {symbol} - SKIPPING' )
        return None, None
    
    with logger.timer( ' -- 5 min data ---' ):
        df_5min, meta_5min = etf_data( symbol=symbol, output='kibot_5min' )
        df_5min = df_5min.assign(
            name=meta_5min['name'],
            asset_class=meta_5min['asset_class'],
            date=df_5min.time.dt.normalize(),
            week=df_5min.time.dt.year * 10000 + df_5min.time.dt.month * 100 + df_5min.time.dt.week,
            month=df_5min.time.dt.year * 10000 + df_5min.time.dt.month * 100
        )
        df_5min = df_5min.rename(
            columns={ c: f'{c}_5min' for c in ['open', 'high', 'low', 'close', 'volume', 'dollar_volume'] }
        )
    with logger.timer( ' -- end of day data---' ):
        df_day, meta_day = etf_data( symbol=symbol, output='kibot_day' )
        df_day = df_day.assign( date=lambda df: df.time.dt.normalize() )
        df_day = (df_day
                  .rename( columns={ c: f'{c}_day' for c in ['open', 'high', 'low', 'close'] } ))
        # add in yesterday's close
        df_day = df_day.merge(
            (df_day
             .set_index( 'date' )['close_day']
             .shift( 1 )
             .reset_index()
             .rename( columns={ 'close_day': 'close_prev_day' } )),
            on='date', how='left',
        )
    with logger.timer( ' -- weekly data --- ' ):
        week_indicator = df_day.time.dt.year * 10000 + df_day.time.dt.month * 100 + df_day.time.dt.week
        df_week = pd.DataFrame( {
            'open_week': df_day.groupby( week_indicator )['open_day'].first(),
            'high_week': df_day.groupby( week_indicator )['high_day'].max(),
            'low_week': df_day.groupby( week_indicator )['low_day'].min(),
            'close_week': df_day.groupby( week_indicator )['close_day'].last(),
        } ).reset_index().rename( columns={ 'time': 'week' } )
        df_week = df_week.merge(
            df_week.set_index( 'week' )['close_week'].shift( 1 ).reset_index().rename(
                columns={ 'close_week': 'close_prev_week' } ),
            on='week', how='left',
        )
    with logger.timer( ' -- monthly data --- ' ):
        month_indicator = df_day.time.dt.year * 10000 + df_day.time.dt.month * 100
        df_month = pd.DataFrame( {
            'open_month': df_day.groupby( month_indicator )['open_day'].first(),
            'high_month': df_day.groupby( month_indicator )['high_day'].max(),
            'low_month': df_day.groupby( month_indicator )['low_day'].min(),
            'close_month': df_day.groupby( month_indicator )['close_day'].last(),
        } ).reset_index().rename( columns={ 'time': 'month' } )
    with logger.timer( 'merging' ):
        df = df_5min
        df = df.merge( df_day.filter_mx( ['date', '/.*_day/'] ), on='date', how='left' )
        df = df.merge( df_week, on='week', how='left' )
        df = df.merge( df_month, on='month', how='left' )
    # --- drop pre and post market hour data ---
    with logger.timer( 'pre and post market adjustment' ):
        df = df.set_index( ['time', 'symbol'] )
        df['pre_market'] = df.time_index.time < dt.time( 9, 30 )
        df['market'] = ((df.time_index.time >= dt.time( 9, 30 ))
                        & (df.time_index.time < dt.time( 16, 0 )))
        df['after_market'] = df.time_index.time >= dt.time( 16, 0 )
    if plots:
        with logger.timer( 'plotting intervals in each session' ):
            figure()
            mkt_df = df[df.market]
            subplot( 131 )
            plot( mkt_df.groupby( mkt_df.time_index.normalize() ).count() )
            gca().set_title( 'num market intervals per day ', alpha=0.5 )
            mkt_df = df[df.pre_market]
            subplot( 132 )
            plot( mkt_df.groupby( mkt_df.time_index.normalize() ).count() )
            gca().set_title( 'num pre-market intervals per day ', alpha=0.5 )
            mkt_df = df[df.after_market]
            subplot( 133 )
            plot( mkt_df.groupby( mkt_df.time_index.normalize() ).count() )
            gca().set_title( 'num after-market intervals per day ', alpha=0.5 )
            disp( 'number of market intervals',
                df[df.market].groupby( df[df.market].time_index.normalize() )['open_5min'].count().max(), h=2 )
    if mkt_hours_only:
        df = df[df.market].drop_columns( '/.*market/' )

    # compute average intraday and intra-weekvolume and dollar volume numbers
    for ft in ['volume', 'dollar_volume']:
        df[f'{ft}_intraday'] = (
            df.groupby( 'date' )[f'{ft}_5min']
                .expanding().mean()
                .reset_index( level=0, drop=True )
        )
        df[f'{ft}_intraweek'] = (
            df.groupby( 'week' )[f'{ft}_5min']
                .expanding().mean()
                .reset_index( level=0, drop=True )
        )
        df[f'{ft}_1hr'] = df.rolling( INTERVALS_PER_HOUR )[f'{ft}_5min'].sum()
        df[f'{ft}_1day'] = df.rolling( INTERVALS_PER_DAY )[f'{ft}_5min'].sum()

    # compute log features
    for ft in mx.Features( ['/.*volume.*/', 'open_5min', ] ).find( df ):
        df[f'log_{ft}'] = df[ft].log()
    with logger.timer( ' intraday and intraweek price volatility' ):
        df['logprice_vol_intraday'] = (
            df.groupby( 'date', group_keys=False )['log_open_5min']
                .expanding().std()
                .reset_index( level=0, drop=True )
        )
        df['logprice_vol_intraweek'] = (
            df.groupby( 'week', group_keys=False )['log_open_5min']
                .expanding().std()
                .reset_index( level=0, drop=True )
        )
    
    with logger.timer( 'adding calendar features' ):
        df = mx.calendar_features( df )
    
    def log_rtn( da1, da2, x, y ):
        return (np.maximum( da1.loc[:, :, x], 1e-6 ).log()
                - np.maximum( da2.loc[:, :, y], 1e-6 ).log())
    
    with logger.timer( '# -- dataarray world ----' ):
        da = df.filter_mx( Number ).to_dataarray_mx()
        # compute various kinds of leading and lagging returns
        da = da.assign_features(
            # intraday returns
            logrtn_lag_intraday=log_rtn( da, da, 'open_5min', 'open_day' ),
            # overnight returns
            logrtn_lag_overnight=log_rtn( da, da, 'open_day', 'close_prev_day' ),
            # returns from start of week
            logrtn_lag_intraweek=log_rtn( da, da, 'open_day', 'open_week' ),
            # weekend returns
            logrtn_lag_weekend=log_rtn( da, da, 'open_week', 'close_prev_week' ),
            # returns from start of month
            logrtn_lag_intramonth=log_rtn( da, da, 'open_day', 'open_month' ),
            # leading return to close of day
            logrtn_lead_intraday=log_rtn( da, da, 'close_day', 'close_5min' ),
            # 5 min leading and lagging returns
            logrtn_lag_5min=log_rtn( da, da.lag( 1 ), 'open_5min', 'open_5min' ),
            logrtn_lead_5min=log_rtn( da.lag( -1 ), da, 'close_5min', 'close_5min' ),
            # 1 hr leading and lagging returns
            logrtn_lag_1hr=log_rtn( da, da.lag( 12 ), 'open_5min', 'open_5min' ),
            logrtn_lead_1hr=log_rtn( da.lag( -12 ), da, 'close_5min', 'close_5min' ),
        )
        # don't know how to compute 1day returns otherwise
        if mkt_hours_only:
            da = da.assign_features(
                logrtn_lag_1day=log_rtn( da, da.lag( INTERVALS_PER_DAY ), 'open_5min', 'open_5min' ),
                logrtn_lead_1day=log_rtn( da.lag( -INTERVALS_PER_DAY ), da, 'close_5min', 'close_5min' ),
            )
    # do a test here
    assert (df.index.levels[1] == [symbol]).all() and (da.symbol == symbol).all().item(), 'symbol fuckup'
    return df, da

########################################################################################
# --- compute returns at 30 min frequency-----
########################################################################################
@mx.memoize
def build_30min_data( symbol, mkt_hours_only=True, plots=False, verbose=False ):
    '''
    Derived from compute_returns. Same logic except that it donsamples to 30mins
    
    Parameters
    ----------
    symbol
    mkt_hours_only
        if True - restrict results to market hours only
    plots
    verbose

    Returns
    -------
    
    
    Test
    ----
    ```
        df, da = eld.build_30min_data( symbol='SPY' )
        
        with mx.timer( 'sanity checking returns' ):
            figure()
            subplot( 221 )
            plot( da.loc[:, :, 'open_30min'].values.reshape( -1 ),
                  da.loc[:, :, 'logrtn_lag_30min'].cumsum().exp().values.reshape( -1 ),
                  '.b', alpha=0.3 )
            gca().set_title( 'cum returns vs open' )
            
            subplot( 222 )
            plot( da.loc[:, :, 'logrtn_lag_30min'].values.reshape( -1 ),
                  da.loc[:, :, 'logrtn_lead_30min'].shift( time=1 ).values.reshape( -1 ), '.' )
            gca().set_title( 'leading vs lagg' )
            
            subplot( 223 )
            plot( da.loc[:, :, 'close_prev_day'].values.reshape( -1 ),
                  da.loc[:, :, 'close_day'].lag( '1D' ).values.reshape( -1 ), '.' )
            gca().set_title( 'curr vs prev day close' )
            
            subplot( 224 )
            plot( da.loc[:, :, 'close_prev_week'].values.reshape( -1 ),
                  da.loc[:, :, 'close_week'].values.reshape( -1 ), '.' )
            gca().set_title( 'curr vs prev week close' )
    ```
    '''
    INTERVALS_PER_HOUR = 2
    INTERVALS_PER_DAY = int( 6.5 * INTERVALS_PER_HOUR )
    logger = mx.Logger( 'compute_returns', verbose=verbose )
    if symbol.startswith( 'INCOMPLETE' ):
        logger.warn( f'No data for {symbol} - SKIPPING' )
        return None, None
    with logger.timer( ' -- 5 min data ---' ):
        df_5min, meta_5min = etf_data( symbol=symbol, output='kibot_5min' )
        # reamp time to be in 30 min increments
        df_5min = (
            df_5min
            .assign( time=lambda df: ( df.time.dt.normalize() + pd.to_timedelta( df.time.dt.hour, 'H' )
                                       + pd.to_timedelta( (df.time.dt.minute > 30) * 30, 'min' ) ) )
        )
    with logger.timer( '# downsample to 30mins'):
        df_30min = (
            pd.concat(
                [
                    df_5min.groupby( 'time' ).open.first(),
                    df_5min.groupby( 'time' ).high.max(),
                    df_5min.groupby( 'time' ).low.min(),
                    df_5min.groupby( 'time' ).close.last(),
                    df_5min.groupby( 'time' ).volume.sum(),
                    df_5min.groupby( 'time' ).dollar_volume.sum(),
                ],
                axis=1
            )
            .reset_index()
            .assign(
                symbol=meta_5min['symbol'],
                asset_class=meta_5min['asset_class'],
                name=meta_5min['name'],
                date=lambda df: df.time.dt.normalize(),
                week=lambda df: (df.time.dt.year * 10000 + df.time.dt.month * 100 + df.time.dt.week),
                month=lambda df: (df.time.dt.year * 10000 + df.time.dt.month * 100),
            )
            .rename( columns={ c: f'{c}_30min' for c in ['open', 'high', 'low', 'close', 'volume', 'dollar_volume'] } )
        )
    with logger.timer( ' -- end of day data---' ):
        df_day, meta_day = etf_data( symbol=symbol, output='kibot_day' )
        df_day = df_day.assign( date=lambda df: df.time.dt.normalize() )
        df_day = df_day.rename( columns={ c: f'{c}_day' for c in ['open', 'high', 'low', 'close'] } )
        # add in yesterday's close
        df_day = df_day.merge(
            (df_day
             .set_index( 'date' )['close_day']
             .shift( 1 )
             .reset_index()
             .rename( columns={ 'close_day': 'close_prev_day' } )),
            on='date', how='left',
        )
    with logger.timer( ' -- weekly data --- ' ):
        week_indicator = df_day.time.dt.year * 10000 + df_day.time.dt.month * 100 + df_day.time.dt.week
        df_week = pd.DataFrame( {
            'open_week': df_day.groupby( week_indicator )['open_day'].first(),
            'high_week': df_day.groupby( week_indicator )['high_day'].max(),
            'low_week': df_day.groupby( week_indicator )['low_day'].min(),
            'close_week': df_day.groupby( week_indicator )['close_day'].last(),
        } ).reset_index().rename( columns={ 'time': 'week' } )
        df_week = df_week.merge(
            df_week.set_index( 'week' )['close_week'].shift( 1 ).reset_index().rename(
                columns={ 'close_week': 'close_prev_week' } ),
            on='week', how='left',
        )
    with logger.timer( ' -- monthly data --- ' ):
        month_indicator = df_day.time.dt.year * 10000 + df_day.time.dt.month * 100
        df_month = pd.DataFrame( {
            'open_month': df_day.groupby( month_indicator )['open_day'].first(),
            'high_month': df_day.groupby( month_indicator )['high_day'].max(),
            'low_month': df_day.groupby( month_indicator )['low_day'].min(),
            'close_month': df_day.groupby( month_indicator )['close_day'].last(),
        } ).reset_index().rename( columns={ 'time': 'month' } )
    with logger.timer( 'merging' ):
        df = df_30min
        df = df.merge( df_day.filter_mx( ['date', '/.*_day/'] ), on='date', how='left' )
        df = df.merge( df_week, on='week', how='left' )
        df = df.merge( df_month, on='month', how='left' )
    # --- drop pre and post market hour data ---
    with logger.timer( 'pre and post market determination' ):
        df = df.set_index( ['time', 'symbol'] )
        df['pre_market'] = df.time_index.time < dt.time( 9, 30 )
        df['market'] = ((df.time_index.time >= dt.time( 9, 30 )) & (df.time_index.time < dt.time( 16, 0 )))
        df['after_market'] = df.time_index.time >= dt.time( 16, 0 )
        if mkt_hours_only:
            df = df[df.market].drop_columns( '/.*market/' )
    with logger.timer( '# compute intraday and intra-week volume and dollar volume' ):
        for ft in ['volume', 'dollar_volume']:
            df[f'{ft}_intraday'] = (
                df.groupby( 'date' )[f'{ft}_30min']
                    .expanding().sum()
                    .reset_index( level=0, drop=True )
            )
            df[f'{ft}_intraweek'] = (
                df.groupby( 'week' )[f'{ft}_30min']
                    .expanding().sum()
                    .reset_index( level=0, drop=True )
            )
            df[f'{ft}_1hr'] = df.rolling( INTERVALS_PER_HOUR )[f'{ft}_30min'].sum()
            df[f'{ft}_1day'] = df.rolling( INTERVALS_PER_DAY )[f'{ft}_30min'].sum()
    with logger.timer( 'adding calendar features' ):
        df = mx.calendar_features( df )
    with logger.timer( '# -- dataarray world ----' ):
        da = df.to_dataarray_mx( features=[float, int] )
        # compute various kinds of leading and lagging returns
        log_rtn = lambda da1, da2, x, y: (np.maximum( da1.loc[:, :, x], 1e-6 ).log()
                                          - np.maximum( da2.loc[:, :, y], 1e-6 ).log())
        da = da.assign_features(
            # intraday returns
            logrtn_lag_intraday=log_rtn( da, da, 'open_30min', 'open_day' ),
            # overnight returns
            logrtn_lag_overnight=log_rtn( da, da, 'open_day', 'close_prev_day' ),
            # returns from start of week
            logrtn_lag_intraweek=log_rtn( da, da, 'open_day', 'open_week' ),
            # weekend returns
            logrtn_lag_weekend=log_rtn( da, da, 'open_week', 'close_prev_week' ),
            # returns from start of month
            logrtn_lag_intramonth=log_rtn( da, da, 'open_day', 'open_month' ),
            # leading return to close of day
            logrtn_lead_intraday=log_rtn( da, da, 'close_day', 'close_30min' ),
            # 5 min leading and lagging returns
            logrtn_lag_30min=log_rtn( da, da.lag( 1 ), 'open_30min', 'open_30min' ),
            logrtn_lead_30min=log_rtn( da.lag( -1 ), da, 'close_30min', 'close_30min' ),
            # 1 hr leading and lagging returns
            logrtn_lag_1hr=log_rtn( da, da.lag( INTERVALS_PER_HOUR ), 'open_30min', 'open_30min' ),
            logrtn_lead_1hr=log_rtn( da.lag( -INTERVALS_PER_HOUR ), da, 'close_30min', 'close_30min' ),
        )
        if mkt_hours_only:
            # don't know how to compute 1day returns otherwise
            da = da.assign_features(
                logrtn_lag_1day=log_rtn( da, da.lag( INTERVALS_PER_DAY ), 'open_30min', 'open_30min' ),
                logrtn_lead_1day=log_rtn( da.lag( -INTERVALS_PER_DAY ), da, 'close_30min', 'close_30min' ),
            )
    return da, meta_5min



def build_and_save_30min_data( save_path='/home/fj/experiments/etf_technical/kibot_etf_30min_data', verbose=True ):
    '''run the build 30min data above - save the data to bz2 files, and then delete all checkpoints
    related to etf_data, compute_return and build_30min_data and zip up all the original kibot data
    '''
    logger = mx.Logger( 'build_features', verbose=verbose )
    with logger.timer( 'loading the data'):
        for symbol in logger.pbar( kibot_day_files.keys() ):
            if symbol.startswith( 'INCOMPLETE' ):
                logger.warn( f'No data for {symbol} - SKIPPING' )
                continue
            logger.info( 'loading data for ', symbol )
            da, meta = build_30min_data( symbol=symbol )
            da.to_netcdf( os.path.join( save_path, f'{symbol}.netcdf' ) )
            with open( os.path.join( save_path, f'{symbol}.meta' ), 'wb' ) as fp:
                pickle.dump( meta, fp )
    return
    
KIBOT_TOP200_ETF_DATA_FROM_FEB2020_PROCESSED = '/local/data/kibot/top200_etfs_processed/'

def get_saved_30min_data( symbol, save_path=KIBOT_TOP200_ETF_DATA_FROM_FEB2020_PROCESSED , verbose=True  ):
    '''
    load the data saved in build_and_save_30min_data
    '''
    logger = mx.Logger( 'build_features', verbose=verbose )
    with logger.timer( 'loading data for ', symbol ):
        da = xa.load_dataarray( os.path.join( save_path, f'{symbol}.netcdf' ))
        with open( os.path.join( save_path, f'{symbol}.meta' ), 'rb' ) as fp:
            meta = pickle.load( fp )
    return da, meta

get_saved_30min_data.all_symbols = [
    file.split('/')[-1].replace('.netcdf','')
    for file in sorted( glob.glob( KIBOT_TOP200_ETF_DATA_FROM_FEB2020_PROCESSED + '/*.netcdf' ) )
]




################################################################################################
# build features for the 5min interval online linear model
################################################################################################
@mx.memoize( verbose=True )
def build_features( symbol, verbose=False, haar=False, dilation=12, levels=4,
                    zscore=False, winsorize=0.05, volatility_features=None, returns_features=None,
                    calendar_features=None, volume_features=None, ):
    logger = mx.Logger( 'build_features', verbose=verbose )
    with logger.timer( 'compute_returns' ):
        df, da = compute_returns( symbol=symbol )
    # 1d, 10d and 30d price and return volatility
    with logger.timer( 'compute volatility' ):
        vol = mx.transforms.rolling( windows=np.array( [1, 10, 30] ) * 78, sid='symbol' ).sd()
        vol.features = ['log_open_5min', 'logrtn_lag_5min']
        da = vol( da )
        volatility_features = volatility_features + vol.output_features
    # compute haar transform
    if haar:
        with logger.timer( 'compute haar' ):
            haar = (mx.transforms.haar( levels=levels, dilation=dilation, sid='symbol' )
                    .set( verbose=verbose ))
            da = haar( da, features=['log_dollar_volume_5min', 'logrtn_lag_5min'] )
            haar_features = haar.output_features
    else:
        haar_features = []
    features = volume_features + calendar_features + returns_features + volatility_features + haar_features
    # winsorize the features
    if winsorize:
        with logger.timer( 'winsorizing' ):
            wins = mx.transforms.batch( sid='symbol' ).winsorize( quantiles=winsorize )
            da = wins( da, features=features )
            # drop the original features and rename the winsorized one
            rename_map = dict( zip( wins.output_features, wins.features ) )
            da = da.drop_coords( features=features ).rename_coords( features=rename_map )
    # zscore the features
    if zscore:
        with logger.timer( 'zscoring' ):
            zscore = mx.transforms.rolling( windows=zscore, sid='symbol' ).zscore()
            da = zscore( da, features=features )
            # drop the original features and rename the zscored one
            rename_map = dict( zip( zscore.output_features, zscore.features ) )
            da = da.drop_coords( features=features ).rename_coords( features=rename_map )
            # add a dummy weight column
    da = da.assign_features( weight=da.loc[:, :, returns_features[0]].isfinite().astype( float ) )
    return df, da, features


