# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:04:06 2020

@author: zhouxiyao
"""

import numpy as np

dtype_tick = [('symbol', '<U2'), 
              ('main_symbol', '<U6'),
              ('date', '<M8[ns]'), 
              ('time', '<M8[ns]'), 
              ('last', '<f8'), 
              ('limitDown', '<f8'), 
              ('limitUp', '<f8'), 
              ('curVol', '<i8'), 
              ('askPrice1', '<f8'), 
              ('bidPrice1', '<f8'), 
              ('askVolume1', '<i8'), 
              ('bidVolume1', '<i8'), 
              ('timestamp', '<M8[ns]')]

dtype_bar = np.dtype({'names':['symbol','main_symbol','date', 'time', 'timestamp',
                               'switching', 'ibar_d', 'ibar_m', 'open', 'high', 'low', 'close', 'volume', 'oi',
                               'open_new','high_new','low_new','close_new','volume_new','oi_new'],
                      'formats':['<U2', '<U6', '<M8[ns]', '<M8[ns]', '<M8[ns]',
                                 '?', '<i8', '<i8', '<f8', '<f8', '<f8', '<f8', '<i8', '<i8',
                                 '<f8', '<f8', '<f8', '<f8', '<i8', '<i8']})

dtype_config = [('underlying', '<U6'),
                ('unit', '<i8'),
                ('slip', '<i8'),
                ('commtype', '?'), #手续费类型 True为按金额 False为按手数
                ('comm', '<f8'),
                ('disc', '<f8'), #平今折扣率
                ('minmove', '<f8'),
                ('margin', '<f8')]

dtype_book = [('idx', '<i8'),#相当于索引
              ('count', '<i8'), #连续下单记号
              ('direction', '<i8'), #方向
              ('position', '<i8'), #手数
              ('active', '?'), #该笔交易是否还在进行 True则还在进行 False则已平仓
              ('indate', '<M8[ns]'),
              ('intime', '<M8[ns]'),
              ('outdate', '<M8[ns]'),
              ('outtime', '<M8[ns]'),
              ('entry', '<f8'),
              ('exit', '<f8'),
              ('pnl', '<f8'),
              ('cost', '<f8'),
              ('high', '<f8'),# draw = (last - high)/high or (low - last)/low, depending on direction
              ('low', '<f8')] # maxdraw = min(maxdraw, draw)
              

#%%
config_ag = np.array([('AG', #品种
                       15, #单位
                       1, #滑点跳数
                       1, #表示按金额百分比收取手续费
                       0.00005, #手续费率
                       1.0, #平今折扣率100%
                       1.0, #最小变动价位
                       0.12)], #保证金率
                     dtype=dtype_config)

config_t = np.array([('T', #品种
                      10000, #单位
                      1, #滑点跳数
                      0, #表示按手数收取手续费
                      3.0, #每手手续费
                      0.0, #平今不收
                      0.005, #最小变动价位
                      0.02)], #保证金率
                     dtype=dtype_config)

# %%
tradeconfig = dict(zip(['AG', 'T'],[config_ag, config_t]))
