# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:19:14 2020

@author: zhouxiyao
"""

import numpy as np
from numba import jit

# %%

#六个交易函数 一个跟踪函数
#long买入开仓 sellall全部卖出平仓 sell单笔卖出平仓
#short卖出开仓 coverall全部买入平仓 cover单笔买入平仓
#track跟踪入场后新高新低

#注意: 
#试图以bool索引修改structured array时,
#一定要按 fp['last'][fp['time']<end] += 1 的顺序
#因为fp[fp['time']<end]['last'] 是一个copy的view 在上面修改没有任何意义
#按整数或切片索引是可以的 而且效率更高 如fp[0]['last'] += 1

#做多 若有空头则全平反手
#@jit(nopython=True)
def long(lot,price,t,curidx,count,book,data,config): #idx很关键 初始idx=0 这样首次开仓对应idx=0
    
    #入场价 对盘+滑点
    entry = round_up(price + config[0]['slip'] * config[0]['minmove'], config[0]['minmove'])
    
    #若当前有交易未平仓
    if (count > 0):
        
        #且为空头 则反手
        if (book[curidx]['direction']) == -1:
            #平所有空仓 此处curidx已经+1
            count, curidx = coverall(price,t,curidx,count,book,data,config)
            
        #若当前count对应交易且为多头 则为加仓
        elif book[curidx]['direction'] == 1:
            #指向book下一空行
            curidx += 1
            #更新之前交易的high和low
            book['high'][book['active']] = np.maximum(book[book['active']]['high'], data[t]['close'])
            book['low'][book['active']] = np.minimum(book[book['active']]['low'], data[t]['close'])
        
    #开仓成本
    if config[0]['commtype'] == 1:
        book[curidx]['cost'] = entry * lot * config[0]['unit'] * config[0]['comm']
    else:
        book[curidx]['cost'] = lot * config[0]['comm']
    
    #更新book
    book[curidx]['position'] = lot
    book[curidx]['direction'] = 1
    book[curidx]['indate'] = data[t]['date']
    book[curidx]['intime'] = data[t]['timestamp']
    book[curidx]['entry'] = entry
    book[curidx]['high'] = entry
    book[curidx]['low'] = entry
    book[curidx]['active'] = True
    
    #下单计数加1
    count += 1
    book[curidx]['count'] = count
    
    return count, curidx

#多头全部平仓
#@jit(nopython=True)
def sellall(price,t,curidx,count,book,data,config):
    
    
    if (book[curidx]['direction'] <= 0) or (count == 0): #若当前没有多头 或已经平仓 则不作任何操作
        return count, curidx
    
    else:
        
        exit_ = round_down(price - config[0]['slip'] * config[0]['minmove'], config[0]['minmove'])
                
        #按金额收手续费
        if config[0]['commtype']:
            #平今
            book['cost'][(book['active'])&(book['indate']==data[t]['date'])] += exit_ * book[(book['active'])&(book['indate']==data[t]['date'])]['position'] * config[0]['unit'] * config[0]['comm'] * config[0]['disc']
            #平昨
            book['cost'][(book['active'])&(book['indate']!=data[t]['date'])] += exit_ * book[(book['active'])&(book['indate']!=data[t]['date'])]['position'] * config[0]['unit'] * config[0]['comm']
        else:
        #按手数收手续费
            book['cost'][(book['active'])&(book['indate']==data[t]['date'])] += book[(book['active'])&(book['indate']==data[t]['date'])]['position'] * config[0]['comm'] * config[0]['disc']
            book['cost'][(book['active'])&(book['indate']!=data[t]['date'])] += book[(book['active'])&(book['indate']!=data[t]['date'])]['position'] * config[0]['comm']
            
        #结算前笔做多交易最后一tick的盈亏 pnl不包括交易成本
        book['pnl'][book['active']] = book[book['active']]['position']\
            * (exit_ - book[book['active']]['entry']) * config[0]['unit']\
                - book['cost'][book['active']]
        
        book['outdate'][book['active']] = data[t]['date']
        book['outtime'][book['active']] = data[t]['timestamp']
        book['exit'][book['active']] = exit_
            
        #入场后最高价最低价更新 包括未平仓的
        book['high'][(book['active'])] = np.maximum(book[(book['active'])]['high'], exit_)
        book['low'][(book['active'])] = np.minimum(book[(book['active'])]['low'], exit_)
        #平仓的active=False
        book['active'][book['active']] = False
        #全平 持仓笔数归零
        count = 0
        #curidx指向book下一空行用于开新仓
        curidx += 1
        
        return count, curidx
    
#指定idx卖出平仓 idx可通过book[book['active']]['idx']获取当前未平仓合约的全部idx
#@jit(nopython=True)
def sell(idx,price,t,curidx,count,book,data,config):

    if (count==0) or (book[idx]['direction'] <= 0): #若不是多头 或已经平仓 则不作任何操作
        return count, curidx
    
    #无加仓
    else:
        
        exit_ = round_down(price - config[0]['slip'] * config[0]['minmove'], config[0]['minmove'])
        
        #按金额收手续费
        if config[0]['commtype']:
            #平仓成本
            if book[idx]['indate'] == data[t]['date']:
                #平今
                book[idx]['cost'] += exit_ * book[idx]['position'] * config[0]['unit'] * config[0]['comm'] * config[0]['disc']
            else:
                #平昨
                book[idx]['cost'] += exit_ * book[idx]['position'] * config[0]['unit'] * config[0]['comm']
        
        #按手数收手续费        
        else:
            if book[idx]['indate'] == data[t]['date']:
                book[idx]['cost'] += book[idx]['position'] * config[0]['comm'] * config[0]['disc']
            else:
                book[idx]['cost'] += book[idx]['position'] * config[0]['comm']
            
        #结算前笔做多交易最后一tick的盈亏
        book[idx]['pnl'] = book[idx]['position'] * (exit_ - book[idx]['entry']) * config[0]['unit'] - book[idx]['cost']
        
        book[idx]['outdate'] = data[t]['date']
        book[idx]['outtime'] = data[t]['timestamp']
        book[idx]['exit'] = exit_
        book[idx]['high'] = max(book[idx]['high'], exit_)
        book[idx]['low'] = min(book[idx]['low'], exit_)
        #状态:已平仓
        book[idx]['active'] = False
        
        #平一单
        count -= 1
        #若全部平仓 则与sellall相同 curidx+=1
        if count == 0:
            curidx += 1
            
    return count, curidx

#做空 若有空头则全平反手
#@jit(nopython=True)
def short(lot,price,t,curidx,count,book,data,config):
    
    #入场价 对盘+滑点
    entry = round_down(price - config[0]['slip'] * config[0]['minmove'], config[0]['minmove'])
    
    #若当前count对应交易未平仓
    if (book[book['active']].size > 0):
        
        #且为多头 则需结算(所有)未平仓做空交易的盈亏 及平仓成本
        if (book[curidx]['direction']) == 1:
        
            count, curidx = sellall(price,t,curidx,count,book,data,config)
    
        #若当前count对应交易且为空头 则为加仓
        elif book[curidx]['direction'] == -1:
            #指向book下一空行
            curidx += 1
            #更新之前交易的high和low
            book['high'][book['active']] = np.maximum(book[book['active']]['high'], data[t]['close'])
            book['low'][book['active']] = np.minimum(book[book['active']]['low'], data[t]['close'])
            
        
    #开仓成本
    if config[0]['commtype'] == 1:
        book[curidx]['cost'] = entry * lot * config[0]['unit'] * config[0]['comm']
    else:
        book[curidx]['cost'] = lot * config[0]['comm']
    
    #更新book
    book[curidx]['position'] = lot
    book[curidx]['direction'] = -1
    book[curidx]['indate'] = data[t]['date']
    book[curidx]['intime'] = data[t]['timestamp']
    book[curidx]['entry'] = entry
    book[curidx]['high'] = entry
    book[curidx]['low'] = entry
    book[curidx]['active'] = True
    
    #下单计数加1
    count += 1
    book[curidx]['count'] = count
    
    return count, curidx

#空头全部平仓
#@jit(nopython=True)
def coverall(price,t,curidx,count,book,data,config):
    
    
    if (book[curidx]['direction'] >= 0) or (count == 0): #若当前没有空头 或已经平仓 则不作任何操作
        return count, curidx
    
    else:
        
        exit_ = round_up(price + config[0]['slip'] * config[0]['minmove'], config[0]['minmove'])
        
        #按金额收手续费
        if config[0]['commtype']:
            #平今
            book['cost'][(book['active'])&(book['indate']==data[t]['date'])] += exit_ * book[(book['active'])&(book['indate']==data[t]['date'])]['position'] * config[0]['unit'] * config[0]['comm'] * config[0]['disc']
            #平昨
            book['cost'][(book['active'])&(book['indate']!=data[t]['date'])] += exit_ * book[(book['active'])&(book['indate']!=data[t]['date'])]['position'] * config[0]['unit'] * config[0]['comm']
        
        #按手数收手续费
        else:

            book['cost'][(book['active'])&(book['indate']==data[t]['date'])] += book[(book['active'])&(book['indate']==data[t]['date'])]['position'] * config[0]['comm'] * config[0]['disc']
            book['cost'][(book['active'])&(book['indate']!=data[t]['date'])] += book[(book['active'])&(book['indate']!=data[t]['date'])]['position'] * config[0]['comm']
            
            #结算前笔做空交易最后一tick的盈亏
        book['pnl'][book['active']] = book[book['active']]['position']\
            * (book[book['active']]['entry'] - exit_) * config[0]['unit']\
                - book['cost'][book['active']]

        book['outdate'][book['active']] = data[t]['date']                
        book['outtime'][book['active']] = data[t]['timestamp']
        book['exit'][book['active']] = exit_
        
        #入场后最高价最低价更新 包括未平仓的
        book['high'][(book['active'])] = np.maximum(book[(book['active'])]['high'], exit_)
        book['low'][(book['active'])] = np.minimum(book[(book['active'])]['low'], exit_)
        #平仓的active=False
        book['active'][book['active']] = False
            
        #当前单数清零
        count = 0
        #curidx指向book下一空行用于开新仓
        curidx += 1
        
        return count, curidx

#空头单笔平仓
#@jit(nopython=True)
def cover(idx,price,t,curidx,count,book,data,config):
    
    if (count==0) or (book[idx]['direction'] >= 0): #若不是多头 或已经平仓 则不作任何操作
        return count, curidx
    
    #无加仓
    else:
        
        exit_ = round_up(price + config[0]['slip'] * config[0]['minmove'], config[0]['minmove'])
        
        #按金额收手续费
        if config[0]['commtype']:
            #平仓成本
            if book[idx]['indate'] == data[t]['date']:
                #平今
                book[idx]['cost'] += exit_ * book[idx]['position'] * config[0]['unit'] * config[0]['comm'] * config[0]['disc']
            else:
                #平昨
                book[idx]['cost'] += exit_ * book[idx]['position'] * config[0]['unit'] * config[0]['comm']
        
        #按手数收手续费        
        else:
            if book[idx]['indate'] == data[t]['date']:
                book[idx]['cost'] += book[idx]['position'] * config[0]['comm'] * config[0]['disc']
            else:
                book[idx]['cost'] += book[idx]['position'] * config[0]['comm']
            
        #结算前笔做多交易最后一tick的盈亏
        book[idx]['pnl'] = book[idx]['position'] * (book[idx]['entry'] - exit_) * config[0]['unit'] - book[idx]['cost']

        book[idx]['outdate'] = data[t]['date']        
        book[idx]['outtime'] = data[t]['timestamp']
        book[idx]['exit'] = exit_
        book[idx]['high'] = max(book[idx]['high'], exit_)
        book[idx]['low'] = min(book[idx]['low'], exit_)
        #状态:已平仓
        book[idx]['active'] = False
        
        #平一单
        count -= 1
        
        #若全部平仓 则与coverall相同 curidx+=1
        if count == 0:
            curidx += 1
    
    return count, curidx


#什么都不做 更新当前持仓的入场高低点
#@jit(nopython=True)
def track(t,book,data):

    book['high'][(book['active'])] = np.maximum(book[(book['active'])]['high'], data[t]['close'])
    book['low'][(book['active'])] = np.minimum(book[(book['active'])]['high'], data[t]['close'])
    
    return

# %%
#按某个最小变动单位向下取整
#@jit(nopython=True)
def round_down(x,a):
    return np.floor(x/a) * a

#按某个最小变动单位向上取整
#@jit(nopython=True)
def round_up(x,a):
    return np.ceil(x/a) * a