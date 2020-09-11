# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 15:18:01 2020

@author: zhouxiyao
"""

import pandas as pd
import numpy as np
from numba import jit

import sys
sys.path.append('C:/Users/zhouxiyao.TAOLI/Desktop/CTAtick/code')

import tradefunc as tf
import configure as cf
import data_prepare as dp

# %%
#@jit(nopython=True)
def strategy_atr(data, config, length=20, width=2, n=20, add=0.5, cut=0.02, ratio=0.01, principal=1000000.0, max_trade=5000, max_count=4, type_=1):

    #第一步 确定计算哪些指标
    dtype_indicator = np.dtype({'names':['tr','atr','mid','n','upper','lower','tr_new'],
                                'formats':['<f8','<f8','<f8','<f8','<f8','<f8','<f8']})
    indicator = np.zeros(data.size, dtype=dtype_indicator)
    
    #第二步 初始化
    book = np.zeros(max_trade, dtype=cf.dtype_book)
    book['idx'] = np.arange(max_trade) #索引
    curidx = 0
    count = 0
    INVALID = max(length, n) #数据不足以计算指标的期数
    #BARS = data['ibar_d'].max()
    #MINMOVE = config[0]['minmove']
    
    #第一条数据没有tr
    indicator[0] = np.nan
    
    #有tr 无atr
    for t in range(1, INVALID-1):
        
        #非切换后第一bar
        if not data[t-1]['switching']:

            hc = abs(data[t]['high'] - data[t-1]['close'])
            cl = abs(data[t-1]['close'] - data[t]['low'])
            
            hc_new = abs(data[t]['high_new'] - data[t-1]['close_new'])
            cl_new = abs(data[t-1]['close_new'] - data[t]['low_new'])
        
        else:
            
            hc = abs(data[t]['high'] - data[t-1]['close_new'])
            cl = abs(data[t-1]['close_new'] - data[t]['low'])
            
            hc_new = np.nan
            cl_new = np.nan #无回溯数据 但计算指标不会用到这里
            
        hl = abs(data[t]['high'] - data[t]['low'])
        hl_new = abs(data[t]['high_new'] - data[t]['low_new'])
            
        indicator[t]['tr'] = max(hl, hc, cl)
        indicator[t]['tr_new'] = max(hl_new, hc_new, cl_new)
            
        indicator[t]['atr'] = np.nan
        indicator[t]['mid'] = np.nan
        indicator[t]['n'] = np.nan
        indicator[t]['upper'] = np.nan
        indicator[t]['lower'] = np.nan
        
    for t in range(INVALID-1, data.size):
        
        #非切换后第一bar
        if not data[t-1]['switching']:

            hc = abs(data[t]['high'] - data[t-1]['close'])
            cl = abs(data[t-1]['close'] - data[t]['low'])
            
            hc_new = abs(data[t]['high_new'] - data[t-1]['close_new'])
            cl_new = abs(data[t-1]['close_new'] - data[t]['low_new'])
        
        else:
            
            hc = abs(data[t]['high'] - data[t-1]['close_new'])
            cl = abs(data[t-1]['close_new'] - data[t]['low'])
            
            hc_new = np.nan
            cl_new = np.nan #无回溯数据 但计算指标不会用到这里
            
        hl = abs(data[t]['high'] - data[t]['low'])
        hl_new = abs(data[t]['high_new'] - data[t]['low_new'])
        
        indicator[t]['tr'] = max(hl, hc, cl)
        indicator[t]['tr_new'] = max(hl_new, hc_new, cl_new)
        
        #当前是主力切换后第几根bar
        IBAR_M = data[t]['ibar_m']
        
        #正常计算
        if IBAR_M >= length:
            
            indicator[t]['atr'] = np.mean(indicator[t-length+1:t+1]['tr']) #atr
            indicator[t]['mid'] = np.mean(data[t-length+1:t+1]['close']) #中线
        
        #合约切换
        else:
                
            indicator[t]['atr'] = (np.sum(indicator[t-IBAR_M+1 : t+1]['tr'])\
                                   +np.sum(indicator[t-length+1 : t-IBAR_M+1]['tr_new'])) / length
            indicator[t]['mid'] = (np.sum(data[t-IBAR_M+1 : t+1]['close'])\
                                   +np.sum(data[t-length+1 : t-IBAR_M+1]['close_new'])) / length
        #正常计算    
        if IBAR_M >= n:
            
            indicator[t]['n'] = np.mean(indicator[t-n+1:t+1]['tr']) #海龟n值 实际为n期atr
        
        #合约切换
        else:
            
            indicator[t]['n'] = (np.sum(indicator[t-IBAR_M+1 : t+1]['tr'])\
                                   +np.sum(indicator[t-n+1 : t-IBAR_M+1]['tr_new'])) / length
        #上下轨
        indicator[t]['upper'] = indicator[t]['mid'] + width * indicator[t]['atr'] #上轨
        indicator[t]['lower'] = indicator[t]['mid'] - width * indicator[t]['atr'] #下轨
        
        #交易
        #以损定量+浮盈按0.5N加仓
        if type_ == 1:
            #以损定量的下单手数
            LOT = np.floor((principal * ratio) / (config[0]['unit'] * indicator[t]['n']))
        
        
        #移仓换月
        if data[t]['switching']:
            
            #换仓前最新头寸方向 分为1 0 -1三种
            DIRECTION = book[curidx]['direction']
            
            #有多头需要移仓
            if (DIRECTION == 1) and (count>0):
                #当前总持仓
                ALL = book[book['active']]['position'].sum()
                
                #出现平仓信号
                if data[t]['low'] < indicator[t]['mid']:
                    #卖出平仓全部现合约
                    count, curidx = tf.sellall(min(indicator[t]['mid'], data[t]['open']), t,curidx,count,book,data,config)
                    
                #出现反手做空信号
                elif data[t]['low'] < indicator[t]['lower']:
                    #卖出平仓全部现合约
                    count, curidx = tf.sellall(min(indicator[t]['lower'], data[t]['open']), t,curidx,count,book,data,config)
                    #按close_new卖出开仓新合约LOT手
                    if LOT >0:
                        count, curidx = tf.short(LOT, data[t]['close_new'], t,curidx,count,book,data,config)
                    
                #出现加仓信号 价格超过最高入场价的add倍n值,且持仓笔数不满max_count笔
                elif data[t]['high'] > (max(book[book['active']]['entry']) + add * indicator[t]['n']) and count < max_count:
                    #卖出平仓全部现合约
                    count, curidx = tf.sellall(data[t]['close'], t,curidx,count,book,data,config)
                    #按close_new买入开仓新合约LOT+ALL手
                    count, curidx = tf.long(ALL+LOT, data[t]['close_new'], t,curidx,count,book,data,config)
                    
                #出现止损信号 价格下跌点数超过任一笔头寸entry*cut
                elif book[(book['active'])&(data[t]['low']<book['entry']*(1-cut))].size > 0:
                    #计算需平仓手数PART
                    PART = book[(book['active'])&(data[t]['low']<book['entry']*(1-cut))]['position'].sum()
                    #需平仓的持仓索引
                    IDXS = book[(book['active'])&(data[t]['low']<book['entry']*(1-cut))]['idx'] #1darray
                    #分别以止损价平仓
                    for IDX in IDXS:
                        count, curidx = tf.sell(IDX, min(book[IDX]['entry']*(1-cut), data[t]['open']), t,curidx,count,book,data,config)
                    
                    #卖出平仓全部现合约
                    count, curidx = tf.sellall(data[t]['close'], t,curidx,count,book,data,config)
                    #按close_new移仓ALL-PART手
                    if PART < ALL:
                        count, curidx = tf.long(ALL-PART,data[t]['close_new'], t,curidx,count,book,data,config)
                    
                #普通移仓
                else:
                    #卖出平仓全部现合约
                    count, curidx = tf.sellall(data[t]['close'], t,curidx,count,book,data,config)
                    #按close_new移仓
                    count, curidx = tf.long(ALL,data[t]['close_new'], t,curidx,count,book,data,config)
                    
            #有空头需要移仓
            elif (DIRECTION == -1) and (count>0):
                #当前总持仓
                ALL = book[book['active']]['position'].sum()
                
                #出现平仓信号
                if data[t]['high'] > indicator[t]['mid']:
                    #买入平仓全部现合约
                    count, curidx = tf.coverall(max(indicator[t]['mid'], data[t]['open']), t,curidx,count,book,data,config)
                    
                #出现反手做多信号
                elif data[t]['high'] > indicator[t]['upper']:
                    #买入平仓全部现合约
                    count, curidx = tf.coverall(max(indicator[t]['upper'], data[t]['open']), t,curidx,count,book,data,config)
                    #按close_new买入开仓新合约LOT手
                    if LOT > 0:
                        count, curidx = tf.long(LOT, data[t]['close_new'], t,curidx,count,book,data,config)
                    
                #出现加仓信号 价格跌过最低入场价的add倍n值,且持仓笔数不满max_count笔
                elif data[t]['low'] < (min(book[book['active']]['entry']) - add * indicator[t]['n']) and count < max_count:
                    #买入平仓全部现合约
                    count, curidx = tf.coverall(data[t]['close'], t,curidx,count,book,data,config)
                    #按close_new卖出开仓新合约LOT+ALL手
                    count, curidx = tf.short(ALL+LOT, data[t]['close_new'], t,curidx,count,book,data,config)
                    
                #出现止损信号 价格上涨点数超过任一笔头寸entry*cut
                elif book[(book['active'])&(data[t]['high']>book['entry']*(1+cut))].size > 0:
                    #计算需平仓手数PART
                    PART = book[(book['active'])&(data[t]['high']>book['entry']*(1+cut))]['position'].sum()
                    #需平仓的持仓索引
                    IDXS = book[(book['active'])&(data[t]['high']>book['entry']*(1+cut))]['idx'] #1darray
                    #分别以止损价平仓
                    for IDX in IDXS:
                        count, curidx = tf.cover(IDX, max(book[IDX]['entry']*(1+cut), data[t]['open']), t,curidx,count,book,data,config)
                    
                    #按close卖出平仓剩余全部现合约
                    count, curidx = tf.coverall(data[t]['close'], t,curidx,count,book,data,config)
                    #按close_new移仓ALL-PART手
                    if PART < ALL:
                        count, curidx = tf.short(ALL-PART,data[t]['close_new'], t,curidx,count,book,data,config)
                    
                #普通移仓
                else:
                    #卖出平仓全部现合约
                    count, curidx = tf.coverall(data[t]['close'], t,curidx,count,book,data,config)
                    #按close_new移仓ALL-PART手
                    count, curidx = tf.short(ALL,data[t]['close_new'], t,curidx,count,book,data,config)          
            
            #无头寸需移仓 只看是否开仓
            else:
                #出现买入开仓信号
                if data[t]['high'] > indicator[t]['upper']:
                    #买入新合约
                    if LOT > 0:
                        count, curidx = tf.long(LOT, data[t]['close_new'], t,curidx,count,book,data,config)
                
                #出现卖出开仓信号
                elif data[t]['low'] < indicator[t]['lower']:
                    #卖出新合约
                    if LOT > 0:
                        count, curidx = tf.short(LOT, data[t]['close_new'], t,curidx,count,book,data,config)
                
                #无事发生
                else:
                    continue
        #非移仓换月        
        else:
            #最新头寸方向 分为1 0 -1三种
            DIRECTION = book[curidx]['direction']
            
            #有多头
            if (DIRECTION == 1) and (count>0):
                
                #出现平仓信号
                if data[t]['low'] < indicator[t]['mid']:
                    #卖出平仓全部现合约
                    count, curidx = tf.sellall(min(indicator[t]['mid'], data[t]['open']), t,curidx,count,book,data,config)
                    
                #出现反手做空信号
                elif data[t]['low'] < indicator[t]['lower']:
                    #反手做空LOT手
                    if LOT > 0:
                        count, curidx = tf.short(LOT, min(indicator[t]['lower'],data[t]['open']), t,curidx,count,book,data,config)
                    
                #出现加仓信号 价格超过最高入场价的add倍n值,且持仓笔数不满max_count笔
                elif data[t]['high'] > (max(book[book['active']]['entry']) + add * indicator[t]['n']) and count < max_count:
                    #买入开仓 加仓LOT手
                    if LOT > 0:
                        count, curidx = tf.long(LOT, max((max(book[book['active']]['entry'])+add*indicator[t]['n']),data[t]['open']), t,curidx,count,book,data,config)
                    
                #出现止损信号 价格下跌点数超过任一笔头寸entry*cut
                elif book[(book['active'])&(data[t]['low']<book['entry']*(1-cut))].size > 0:
                    #需平仓的持仓索引
                    IDXS = book[(book['active'])&(data[t]['low']<book['entry']*(1-cut))]['idx'] #1darray
                    #分别以止损价平仓
                    for IDX in IDXS:
                        count, curidx = tf.sell(IDX, min(book[IDX]['entry']*(1-cut), data[t]['open']), t,curidx,count,book,data,config)
                    
                #无信号
                else:
                    #跟进每笔持仓的新高新低
                    tf.track(t, book, data)
                    
            #有空头
            elif (DIRECTION == -1) and (count>0):
                
                #出现平仓信号
                if data[t]['high'] > indicator[t]['mid']:
                    #卖出平仓全部现合约
                    count, curidx = tf.coverall(max(indicator[t]['mid'], data[t]['open']), t,curidx,count,book,data,config)
                    
                #出现反手做多信号
                elif data[t]['high'] > indicator[t]['upper']:
                    #反手做多LOT手
                    if LOT > 0:
                        count, curidx = tf.long(LOT, max(indicator[t]['upper'], data[t]['open']), t,curidx,count,book,data,config)
                    
                #出现加仓信号 价格跌过最低入场价的add倍n值,且持仓笔数不满max_count笔
                elif data[t]['low'] < (min(book[book['active']]['entry']) - add * indicator[t]['n']) and count < max_count:
                    #卖出开仓 加仓LOT手
                    if LOT > 0:
                        count, curidx = tf.short(LOT, min((min(book[book['active']]['entry'])+add*indicator[t]['n']),data[t]['open']), t,curidx,count,book,data,config)
                    
                #出现止损信号 价格上涨点数超过任一笔头寸entry*cut
                elif book[(book['active'])&(data[t]['high']>book['entry']*(1+cut))].size > 0:
                    #需平仓的持仓索引
                    IDXS = book[(book['active'])&(data[t]['high']>book['entry']*(1+cut))]['idx'] #1darray
                    #分别以止损价平仓
                    for IDX in IDXS:
                        count, curidx = tf.sell(IDX, max(book[IDX]['entry']*(1+cut), data[t]['open']), t,curidx,count,book,data,config)
                    
                #无信号
                else:
                    #跟进每笔持仓的新高新低
                    tf.track(t, book, data)
            
            #无头寸 只看是否开仓
            else:
                #出现买入开仓信号
                if data[t]['high'] > indicator[t]['upper']:
                    #买入
                    if LOT > 0:
                        count, curidx = tf.long(LOT, max(indicator[t]['upper'], data[t]['open']), t,curidx,count,book,data,config)
                
                #出现卖出开仓信号
                elif data[t]['low'] < indicator[t]['lower']:
                    #卖出
                    if LOT > 0:
                        count, curidx = tf.short(LOT, min(indicator[t]['lower'], data[t]['open']), t,curidx,count,book,data,config)
                
                #无事发生
                else:
                    continue        
        
    return book, indicator

# %%
df_T_15m = dp.load_future_main(['T'],'2015.03.20','2020.05.31','15')
df_T_15m_withnext = dp.append_next_main(df_T_15m, '15')

T_15m = np.array(list(df_T_15m_withnext.apply(lambda x: tuple(x), axis=1)), dtype = cf.dtype_bar)

CONFIG = cf.tradeconfig['T']

book, indicator = strategy_atr(T_15m,CONFIG)
