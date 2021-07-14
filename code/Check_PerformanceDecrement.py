# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:15:30 2021

@author: ayumu
"""

#%%
import glob
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy import stats
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import math
import scipy.io
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
import numpy.matlib as mb
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
lr = LinearRegression()
Z = norm.ppf

def SDT(hits, misses, fas, crs): # hits, misses, false alarms, correct rejection
    """ returns a dict with d-prime measures given hits, misses, false alarms, and correct rejections"""
    # Floors an ceilings are replaced by half hits and half FA's
    half_hit = 0.5 / (hits + misses)
    half_fa = 0.5 / (fas + crs)
 
    # Calculate hit_rate and avoid d' infinity
    hit_rate = hits / (hits + misses)
    if hit_rate == 1: 
        hit_rate = 1 - half_hit
    if hit_rate == 0: 
        hit_rate = half_hit
 
    # Calculate false alarm rate and avoid d' infinity
    fa_rate = fas / (fas + crs)
    if fa_rate == 1: 
        fa_rate = 1 - half_fa
    if fa_rate == 0: 
        fa_rate = half_fa
 
    # Return d', beta, c and Ad'
    out = {}
    out['d'] = Z(hit_rate) - Z(fa_rate)
    out['beta'] = math.exp((Z(fa_rate)**2 - Z(hit_rate)**2) / 2)
    out['c'] = -(Z(hit_rate) + Z(fa_rate)) / 2
    out['Ad'] = norm.cdf(out['d'] / math.sqrt(2))
    
    return(out)

def CalculateEffect(data1,data2):
    n1 = len(data1)
    n2 = len(data2)
    x1 = np.mean(data1)
    x2 = np.mean(data2)
    s1 = np.std(data1)
    s2 = np.std(data2)
    sd = np.sqrt((s1**2+s2**2)/2)
    s = np.std(data1-data2)
    
    g = abs(x1-x2)/sd
    # g = abs(x1-x2)/s
    biasFac = np.sqrt((n1-2)/(n1-1))
    g_unbiased = g*biasFac
    return g_unbiased

##############
# parameters #
##############
dummy = 0
save_flag = 1
sns.set_context('poster')
project = 'TestMyBrain'

top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/'
source_dir = 'C:/Users/ayumu/Dropbox/gradCPT/data/%s/' %project

fig_dir = '%s/fig/%s/behavior/' %(top_dir,project)
if os.path.isdir(fig_dir)==False: os.mkdir(fig_dir)

out_dir = '%s/behavior/' %source_dir
if os.path.isdir(out_dir)==False: os.mkdir(out_dir)

isi=0.8
mat = scipy.io.loadmat(source_dir + '/TMB_TrialData4Mike.mat')
Trial = mat['TMB_list'][:,2]
sub_num = len(mat['cleanrt'])
df = pd.read_csv(out_dir + 'participants.csv')
df['group'] = np.where(df.Skewness>0,'Over','Under')

DATA = pd.DataFrame()
DATA['trial_type'] = mb.repmat(np.where(Trial==1,'mountain','city'),1,sub_num)[0]
DATA['response'] = mat['cleanresponse'].flatten()
DATA['CorrectOmission'] = np.where((DATA['response']==1)&(DATA['trial_type']=='mountain'),1,0)
DATA['CorrectCommission'] = np.where((DATA['response']==1)&(DATA['trial_type']=='city'),1,0)
DATA['CommissionError'] = np.where(DATA['response']==-1,1,0)
DATA['OmissionError'] = np.where(DATA['response']==0,1,0)
DATA['ReactionTime'] =mat['cleanrt'].flatten()/1000
DATA['outlier'] = np.where((DATA['ReactionTime']>1.36)|(DATA['ReactionTime']<0.24),1,0)
subs = ['sub-%s'%str(sub_i).zfill(5) for sub_i in range(sub_num)]
DATA['subid'] = mb.repmat(subs,300,1).T.flatten()
DATA['trial'] = np.arange(0,sub_num*300,1) %300 +1
DATA['trial_timing'] = np.zeros(sub_num*300)
DATA['trial_timing'] = np.where(DATA['trial']<76,'1st',DATA['trial_timing'])
DATA['trial_timing'] = np.where((DATA['trial']<151)&(DATA['trial']>75),'2nd',DATA['trial_timing'])
DATA['trial_timing'] = np.where((DATA['trial']<226)&(DATA['trial']>150),'3rd',DATA['trial_timing'])
DATA['trial_timing'] = np.where((DATA['trial']<301)&(DATA['trial']>225),'4th',DATA['trial_timing'])

df_sub_sum = DATA.groupby(['subid','trial_timing']).sum().reset_index()
df_sub_sum['trial_timing_num'] = np.zeros(sub_num*4)
df_sub_sum['trial_timing_num'] = np.where(df_sub_sum['trial_timing']=="1st",1,df_sub_sum['trial_timing_num'])
df_sub_sum['trial_timing_num'] = np.where(df_sub_sum['trial_timing']=="2nd",2,df_sub_sum['trial_timing_num'])
df_sub_sum['trial_timing_num'] = np.where(df_sub_sum['trial_timing']=="3rd",3,df_sub_sum['trial_timing_num'])
df_sub_sum['trial_timing_num'] = np.where(df_sub_sum['trial_timing']=="4th",4,df_sub_sum['trial_timing_num'])
lr.fit(df_sub_sum['trial_timing_num'].values.reshape(-1,1),df_sub_sum['dprime'].values.reshape(-1,1))


df_sub = DATA.query('outlier==0&CorrectCommission==1').groupby(['subid']).mean().reset_index()['ReactionTime']
DATA['meanRT'] = mb.repmat(df_sub,300,1).T.flatten()

hits = DATA.groupby(['subid','trial_timing']).sum().reset_index()['CorrectOmission']
misses = DATA.groupby(['subid','trial_timing']).sum().reset_index()['CommissionError']
fas = DATA.groupby(['subid','trial_timing']).sum().reset_index()['OmissionError']
crs = DATA.groupby(['subid','trial_timing']).sum().reset_index()['CorrectCommission']

dprime = []
for hit,miss,fa,cr in zip(hits,misses,fas,crs):
    out = SDT(hit,miss,fa,cr)
    dprime.append(out['d'])

df_sub_sum['dprime'] = dprime
df_sub_sum['meanRT'] = DATA.query('outlier==0').groupby(['subid','trial_timing']).mean().reset_index()['ReactionTime']

import statsmodels.formula.api as smf
model = smf.ols("dprime ~ trial_timing_num  + 1", df_sub_sum).fit().summary()
print(model)

plt.figure(figsize=(8,5))
sns.lineplot(x="trial_timing",y="dprime",data=df_sub_sum)
if save_flag==1:plt.savefig(fig_dir + 'VigilanceDecrement_dprime.pdf', bbox_inches='tight')


df_VTC = DATA.query('outlier==0&CorrectCommission==1')
df_VTC['VTC'] = abs(DATA.query('outlier==0&CorrectCommission==1').groupby(['subid']).ReactionTime.transform(lambda x : zscore(x,ddof=1)))
df_VTC_sub = df_VTC.groupby(['subid','trial_timing']).mean().reset_index()
df_VTC_sub['trial_timing_num'] = np.zeros(sub_num*4)
df_VTC_sub['trial_timing_num'] = np.where(df_VTC_sub['trial_timing']=="1st",1,df_VTC_sub['trial_timing_num'])
df_VTC_sub['trial_timing_num'] = np.where(df_VTC_sub['trial_timing']=="2nd",2,df_VTC_sub['trial_timing_num'])
df_VTC_sub['trial_timing_num'] = np.where(df_VTC_sub['trial_timing']=="3rd",3,df_VTC_sub['trial_timing_num'])
df_VTC_sub['trial_timing_num'] = np.where(df_VTC_sub['trial_timing']=="4th",4,df_VTC_sub['trial_timing_num'])

plt.figure(figsize=(8,5))
sns.lineplot(x="trial_timing",y="VTC",data=df_VTC_sub)
if save_flag==1:plt.savefig(fig_dir + 'VigilanceDecrement_VTC.pdf', bbox_inches='tight')

model = smf.ols("VTC ~ trial_timing_num  + 1", df_VTC_sub).fit().summary()
print(model)

