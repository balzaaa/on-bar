# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 15:02:51 2020

@author: zhouxiyao
"""


import pandas as pd
import numpy as np
import os
from Dolphin_data import GetData
# %%
#读取指数权重数据
def load_weights(weight_path,startdate,enddate,products):
    '''
    Parameters
    ----------
    weight_path : STRING
        the big folder.
    startdate : STRING
        i.e. '2020.02.02'
    enddate : STRING
        i.e. '2020.02.02'
        data of enddate is included.
    products : LIST
        any sub list of [50,300,500].

    Returns
    -------
    weight50_data : DATAFRAME
        000016's date, weights, components.
    weight300_data : DATAFRAME
        000300's date, weights, components.
    weight500_data : DATAFRAME
        000905's date, weights, components.

    '''
    weight50_list = []
    weight300_list = []
    weight500_list = []
    
    for name in os.listdir(weight_path):
        
        path = os.path.join(weight_path, name)

        if ('_weights' in name) and (int(startdate.replace('.',''))<=int(name[:8])<=int(enddate.replace('.',''))):
            
            for filename in os.listdir(path):
                
                if ('.xls' in filename) and ('000016' in filename) and (50 in products):
                    weight50 = pd.read_excel(os.path.join(path,filename),
                                             usecols='A,E,Q',
                                             names=['date','symbol','weight'],
                                             dtype={'symbol':str})
                    weight50['date'] = pd.to_datetime(name[:8],format='%Y-%m-%d')
                    weight50_list.append(weight50)

                elif ('.xls' in filename) and ('000300' in filename) and (300 in products):
                    weight300 = pd.read_excel(os.path.join(path,filename),
                                              usecols='A,E,Q',
                                              names=['date','symbol','weight'],
                                              dtype={'symbol':str})
                    weight300['date'] = pd.to_datetime(name[:8],format='%Y-%m-%d')
                    weight300_list.append(weight300)

                elif ('.xls' in filename) and ('000905' in filename) and (500 in products):
                    weight500 = pd.read_excel(os.path.join(path,filename),
                                              usecols='A,E,Q',
                                              names=['date','symbol','weight'],
                                              dtype={'symbol':str})
                    weight500['date'] = pd.to_datetime(name[:8],format='%Y-%m-%d')
                    weight500_list.append(weight500)
                
                else:
                    continue
                    
        else:
            continue
    if 50 in products:
        weight50_data = pd.concat(weight50_list, ignore_index=True)
    else:
        weight50_data = pd.DataFrame(columns=['date','symbol','weight'])
    
    if 300 in products:
        weight300_data = pd.concat(weight300_list, ignore_index=True)
    else:
        weight300_data = pd.DataFrame(columns=['date','symbol','weight'])
    
    if 500 in products:
        weight500_data = pd.concat(weight500_list, ignore_index=True)
    else:
        weight500_data = pd.DataFrame(columns=['date','symbol','weight'])
    
    return weight50_data, weight300_data, weight500_data

# %%
#读取股票数据
def load_stock(symbols,startdate,enddate,ktype='1'):
    '''
    

    Parameters
    ----------
    symbols : ITERABLE
        stock symbols needed.
    startdate : STRING
        i.e. '2020.02.02'
    enddate : STRING
        i.e. '2020.02.02'
        data of enddate is included.
    ktype : STRING
        i.e. '1' means one minute bar.
        
    Returns
    -------
    stock_data : DATAFRAME
        stock bar data with symbol, date, time, open, high, low, close, volume, turnover.

    '''
    stock_data_list = []
    Get = GetData()
    for symbol in symbols:
        temp = Get.Stock_candle(symbol,startdate,enddate,ktype)
        stock_data_list.append(temp)
        
    stock_data = pd.concat(stock_data_list, ignore_index=True)
    stock_data['time'] = '19700101' + stock_data['date'].dt.strftime('%H%M%S')
    stock_data['time'] = pd.to_datetime(stock_data['time'],format='%Y%m%d%H%M%S')
    stock_data['date'] = stock_data['date'].dt.strftime('%Y-%m-%d')
    
    return stock_data[['symbol','date','time','open','high','low','close','volume','turnover']]

# %%
#按合约读取期货数据
def load_future_contract(symbols,startdate,enddate,ktype='1'):
    '''
    

    Parameters
    ----------
    symbols : ITERABLE
        future symbols needed.
    startdate : STRING
        i.e. '2020.02.02'
    enddate : STRING
        i.e. '2020.02.02'
        data of enddate is included.
    ktype : STRING
        i.e. '1' means one minute bar.

    Returns
    -------
    DATAFRAME
        future bar data.

    '''
    
    future_data_list = []
    Get = GetData()
    for symbol in symbols:
        temp = Get.Future_hist_candle(symbol,startdate,enddate,ktype)
        future_data_list.append(temp)
        if (int(startdate.replace('.',''))<=20180323<=int(enddate.replace('.',''))) and (any ([product in symbol for product in ['IF','IH','IC']])):
            missed = temp[temp['date']=='2018-03-23 09:32:00'].copy()
            missed['date'] -= np.timedelta64(1,'m')
            future_data_list.append(missed)
            
    future_data = pd.concat(future_data_list, ignore_index=True)
    #2.
    future_data['time'] = '19700101' + future_data['date'].dt.strftime('%H%M%S')
    future_data['time'] = pd.to_datetime(future_data['time'],format='%Y%m%d%H%M%S')
    
    #3.
    future_data['timestamp'] = future_data['date']
    
    #1.
    #下面两步用下一日盘填充夜盘的交易日
    future_data.loc[future_data['time']>'1970-01-01 18:00:00','date'] = np.nan
    future_data['date'].bfill(inplace=True)
    #仅保留日期
    future_data['date'] = pd.to_datetime(future_data['date'].dt.date)
    
    return future_data[['product','symbol','date','time','timestamp','open','high','low','close','volume','oi']].sort_values(by=['timestamp'],ignore_index=True).rename(columns={'symbol':'cur_symbol'})

# %%
#按品种读取期货主连数据
def load_future_main(products,startdate,enddate,ktype='1'):
    '''
    

    Parameters
    ----------
    products : ITERABLE
        future products needed. i.e. 'IC'
    startdate : STRING
        i.e. '2020.02.02'
    enddate : STRING
        i.e. '2020.02.02'
        data of enddate is included.
    ktype : STRING
        i.e. '1' means one minute bar.

    Returns
    -------
    DATAFRAME
        future bar data.

    '''
    
    future_data_list = []
    Get = GetData()
    for product in products:
        temp = Get.Future_hist_Mcandle(product,startdate,enddate,'main',ktype)
        future_data_list.append(temp)
        if (int(startdate.replace('.',''))<=20180323<=int(enddate.replace('.',''))) and (product in ['IF','IH','IC']):
            missed = temp[temp['date']=='2018-03-23 09:32:00'].copy()
            missed['date'] -= np.timedelta64(1,'m')
            future_data_list.append(missed)
        
    future_data = pd.concat(future_data_list, ignore_index=True)
    #原始数据date是包括时分秒的timestamp 将其分解为三部分：1.交易日(夜盘+下一日盘为一个交易日) 2.时间(1970-01-01+时分秒) 3.时间戳(真实日期时间)
    
    #2.
    future_data['time'] = '19700101' + future_data['date'].dt.strftime('%H%M%S')
    future_data['time'] = pd.to_datetime(future_data['time'],format='%Y%m%d%H%M%S')
    
    #3.
    future_data['timestamp'] = future_data['date']
    
    #1.
    #下面两步用下一日盘填充夜盘的交易日
    future_data.loc[future_data['time']>'1970-01-01 18:00:00','date'] = np.nan
    future_data['date'].bfill(inplace=True)
    #仅保留日期
    future_data['date'] = pd.to_datetime(future_data['date'].dt.date)
    
    #当前主力的最后一根k线标记为True
    future_data['switching'] = (future_data['main_symbol']!=future_data['main_symbol'].shift(-1)).ffill()
    
    #按日bar计数
    future_data['ibar_d'] = future_data.groupby(['product','date'])['close'].transform(lambda x: np.arange(1,len(x)+1))
    
    #按主力合约bar计数
    future_data['ibar_m'] = future_data.groupby(['product','main_symbol'])['close'].transform(lambda x: np.arange(1,len(x)+1))
    
    return future_data[['product','main_symbol','date','time','timestamp','switching','ibar_d','ibar_m','open','high','low','close','volume','oi']].sort_values(by=['product','timestamp'],ignore_index=True).rename(columns={'product':'symbol'})

# %%
#在主连数据旁边添加下一主力合约数据 用于计算合约切换后的指标
def append_next_main(main_data,ktype='1'):
    #遍历出现过的主力合约及其随后的主力合约 不包括数据范围内最后一个
    next_data_list = []
    for main_now, main_next in zip(main_data['main_symbol'].unique()[:-1], main_data['main_symbol'].unique()[1:]):
        #用于选出对应时间段的下一主力合约数据
        ts_start = main_data[main_data['main_symbol']==main_now]['timestamp'].values[0]
        ts_end = main_data[main_data['main_symbol']==main_now]['timestamp'].values[-1]
        #用于取出main_next数据的范围
        startdate = pd.to_datetime(str(ts_start)).strftime('%Y.%m.%d')
        enddate = pd.to_datetime(str(ts_end)).strftime('%Y.%m.%d')
        
        get_data = load_future_contract([main_next],startdate,enddate,ktype=ktype)
        temp = get_data[(get_data['timestamp']>=ts_start) & (get_data['timestamp']<=ts_end)]
        next_data_list.append(temp)
    
    next_data = pd.concat(next_data_list, ignore_index=True)
    merged = main_data.merge(next_data,how='left',on=['date','time','timestamp'],suffixes=[None,'_new']).ffill() #前向填充nan 由于当前主力没有对应
    
    #由于nan出现过 volume_new和oi_new被转换为float64了 将其转回int64
    #merged['volume_new'] = merged['volume_new'].astype(np.int64)
    #merged['oi_new'] = merged['oi_new'].astype(np.int64)
    
    return merged[['symbol','main_symbol','date','time','timestamp','switching','ibar_d','ibar_m','open','high','low','close','volume','oi','open_new','high_new','low_new','close_new','volume_new','oi_new']]
        
# %%
#按代码读取指数数据
def load_index(symbols,startdate,enddate,ktype='1'):
    '''
    

    Parameters
    ----------
    symbols : ITERABLE
        index symbols needed. i.e. '000300'
    startdate : STRING
        i.e. '2020.02.02'
    enddate : STRING
        i.e. '2020.02.02'
        data of enddate is included.
    ktype : STRING
        i.e. '1' means one minute bar.

    Returns
    -------
    DATAFRAME
        index bar data with product, symbol, date, time, open, high, low, close, volume, turnover.

    '''
    
    index_data_list = []
    Get = GetData()
    for symbol in symbols:
        temp = Get.Stock_index_candle(symbol,startdate,enddate,ktype)
        index_data_list.append(temp)
        
    index_data = pd.concat(index_data_list, ignore_index=True)
    index_data['time'] = '19700101' + index_data['date'].dt.strftime('%H%M%S')
    index_data['time'] = pd.to_datetime(index_data['time'],format='%Y%m%d%H%M%S')
    index_data['timestamp'] = index_data['date']
    index_data['date'] = index_data['date'].dt.strftime('%Y-%m-%d')
    
    #20180131之前 每天有241根分钟线 开盘单独一根 因此将其整合到下一根
    if int(startdate.replace('.',''))<=20180130<=int(enddate.replace('.','')):
        #9:31的开盘价取9:30的
        index_data.loc[(index_data['timestamp']<='2018-01-30 09:31:00')&(index_data['time']=='1970-01-01 09:31:00'),'open']\
            = index_data[(index_data['timestamp']<='2018-01-30 09:31:00')&(index_data['time']=='1970-01-01 09:30:00')]['open'].values
        #成交量和成交额加上9:30的
        index_data.loc[(index_data['timestamp']<='2018-01-30 09:31:00')&(index_data['time']=='1970-01-01 09:31:00'),['volume','turnover']]\
            += index_data[(index_data['timestamp']<='2018-01-30 09:31:00')&(index_data['time']=='1970-01-01 09:30:00')][['volume','turnover']].values
        #删去9:30的
        index_data.drop(index_data[index_data['time']=='1970-01-01 09:30:00'].index, inplace=True)
        
    return index_data[['symbol','date','time','timestamp','open','high','low','close','volume','turnover']].sort_values(by=['symbol','date','time'],ignore_index=True)

# %%
#聚合k线时对不同列的处理
aggfunc_future = {'symbol':'last',
                  'main_symbol':'last',
                  'switching':'last',
                  'date':'last',
                  'time':'last',
                  'timestamp':'last',
                  'open':'first',
                  'high':'max',
                  'low':'min',
                  'close':'last',
                  'volume':'sum',
                  'oi':'last'
                  }
aggfunc_index = {'symbol':'last',
                 'date':'last',
                 'time':'last',
                 'timestamp':'last',
                 'open':'first',
                 'high':'max',
                 'low':'min',
                 'close':'last',
                 'volume':'sum',
                 'turnover':'sum'
                 }
aggfunc_spread = {'date':'last',
                  'time':'last',
                  'timestamp':'last',
                  'open':'first',
                  'high':'max',
                  'low':'min',
                  'close':'last'}

# %%
#聚合N分钟k线 单个产品
def candle_aggregate(data,freq,aggfunc):
    aggr = data.resample(rule='{}min'.format(freq),
                         on='timestamp',
                         closed='right',
                         label='right').agg(aggfunc).dropna().reset_index(drop=True)
    return aggr

#聚合N分钟k线 多个产品合并为一个df
def candle_aggregate_symbols(data,freq,aggfunc):
    agg_list = []
    for symbol in data.symbol.unique(): ##groupby会有bug
        agg_list.append(candle_aggregate(data[data['symbol']==symbol],freq,aggfunc))
    return pd.concat(agg_list,ignore_index=True)