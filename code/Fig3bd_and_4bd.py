# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017 1016

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
from scipy.stats import norm
import scipy.io
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering

Z = norm.ppf

def CheckWhere(num_vol,tr,onset_time,thred):
    onset_time = onset_time + 5 # explain for hemodynamic response
    if onset_time>num_vol*tr: onset_time = num_vol*tr-1 # explain for hemodynamic response
    x = np.arange(tr, round((num_vol+1)*tr,2),tr)
    belong = np.logical_and(onset_time <= x,onset_time>x-thred)
    return belong

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
    biasFac = np.sqrt((n1-2)/(n1-1))
    g_unbiased = g*biasFac
    return g_unbiased
##############
# parameters #
##############
dummy = 0
save_flag = 1
sns.set_context('poster')

# project = 'OriginalGradCPT'
project = 'GradCPT_MindWandering'

top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/'
source_dir = 'C:/Users/ayumu/Dropbox/gradCPT/data/%s/' %project
basin_dir = top_dir + 'data/%s/EnergyLandscape/' %project

roi = 'Schaefer400_8Net'
net_order = ['DefaultMode', 'Limbic', 'PrefrontalControlB', 'PrefrontalControlA','DorsalAttention','Salience','SomatoMotor','Visual']

roi_dir = 'C:/Users/ayumu/Dropbox/gradCPT/Parcellations/%s/' %roi
ROI_files = pd.read_csv(roi_dir + roi + '.csv')
roiname = ROI_files.Network
network = np.unique(roiname)

fig_dir = '%s/fig/%s/behavior/' %(top_dir,project)
if os.path.isdir(fig_dir)==False: os.mkdir(fig_dir)

out_dir = '%s/behavior/' %source_dir
if os.path.isdir(out_dir)==False: os.mkdir(out_dir)

if project == 'GradCPT_MindWandering':
    tr=1.08
    isi=1.3
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants_HC.tsv')[0],delimiter='\t')
    mat = scipy.io.loadmat(basin_dir + roi + '/HC_onlyMW/LocalMin_Summary.mat')
    thred=1.5
    bins=20
elif project == 'OriginalGradCPT':
    tr=2
    isi=0.8
    target = 'IndividualSubject'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants.tsv')[0],delimiter='\t')
    mat = scipy.io.loadmat(basin_dir + roi + '/LocalMin_Summary.mat')
    thred=2
    bins=25
tmp = np.reshape(mat['vectorList'][:,mat['LocalMinIndex']-1],[len(mat['vectorList']),len(mat['LocalMinIndex'])])
brain_activity_pattern = pd.DataFrame(tmp,index=network)
brain_activity_pattern.columns = ['State 1','State 2']
brain_activity_pattern = brain_activity_pattern.reindex(index=net_order)

metric='hamming'
method='complete'
sns.clustermap(brain_activity_pattern, row_cluster=False,
               method=method, metric=metric,
               cbar = False,cmap='Pastel1_r', linewidths=.3)
row_clusters = linkage(pdist(brain_activity_pattern.T,metric=metric),method=method)
row_dendr = dendrogram(row_clusters,no_plot=True)

MyPalette = ["#67a9cf","#ef8a62"]
n_clusters=2
ac = AgglomerativeClustering(n_clusters=n_clusters,
                            affinity=metric,
                            linkage=method)
cluster = ac.fit_predict(brain_activity_pattern.T)

subs = demo['participants_id']
sub_num = len(subs)

# extract signals
DATA = pd.DataFrame()
for sub_i in subs:
    if project == 'GradCPT_MindWandering':
        task_files = glob.glob(source_dir + '/MRI/'+ sub_i +'_task-gradCPTMW*events.tsv');task_files.sort()
        data_files = glob.glob(basin_dir + roi + '/HC_onlyMW/*' + sub_i + '*_BN.csv');data_files.sort()
    elif project == 'GradCPT_reward':
        task_files = glob.glob(source_dir + '/MRI/'+ sub_i +'_task-gradCPT*events.tsv');task_files.sort()
        data_files = glob.glob(basin_dir + roi + '/all/*' + sub_i + '*_BN.csv');data_files.sort()
    else: 
        task_files = glob.glob(source_dir + '/MRI/'+ sub_i +'_task-gradCPT*events.tsv');task_files.sort()
        data_files = glob.glob(basin_dir + roi + '/*' + sub_i + '*_BN.csv');data_files.sort()

    DATA_sub = pd.DataFrame()
    state_run = pd.DataFrame()
    sessions = pd.Series()
    for num_file_i,file_i in enumerate(task_files):
        taskinfo = pd.read_csv(file_i,delimiter='\t')
        data = pd.read_csv(data_files[num_file_i],header=None)
        num_vol = data.shape[0]
        for onset_time in taskinfo['onset']:
            belong = CheckWhere(num_vol,tr,onset_time,thred)
            state_run = state_run.append(data[belong].iloc[0])
            sessions = sessions.append(pd.Series(num_file_i+1))    
        DATA_sub = DATA_sub.append(taskinfo)
    DATA_sub['state'] = state_run.values
    DATA_sub['subid'] = sub_i        
    DATA_sub['session'] = sessions.values        
    DATA = DATA.append(DATA_sub)

DATA['summary_state'] = np.zeros(DATA['state'].shape)
state_all = pd.unique(DATA.state)
state_all.sort()
cluster.sort()
for state_i in state_all:
    DATA.summary_state = np.where(DATA.state == state_i,cluster[int(state_i-1)],DATA.summary_state)

skewness = np.zeros([sub_num,1])
skewness_State1 = np.zeros([sub_num,1])
skewness_State2 = np.zeros([sub_num,1])
params_exg_State1_all = np.zeros([sub_num,3])
params_exg_State2_all = np.zeros([sub_num,3])
params_exg_State1_all_flip = np.zeros([sub_num,3])
params_exg_State2_all_flip = np.zeros([sub_num,3])

for sub_i,sub_name in enumerate(subs):
    skewness[sub_i,:] = DATA.query('CorrectCommission==1&subid==@sub_name')['ReactionTime'].skew()
    skewness_State1[sub_i,:] = DATA.query('CorrectCommission==1&summary_state==0&subid==@sub_name')['ReactionTime'].skew()
    skewness_State2[sub_i,:] = DATA.query('CorrectCommission==1&summary_state==1&subid==@sub_name')['ReactionTime'].skew()
    params_exg_State1_all[sub_i,:] = stats.distributions.exponnorm.fit(DATA.query('CorrectCommission==1&summary_state==0&subid==@sub_name')['ReactionTime'])
    params_exg_State2_all[sub_i,:] = stats.distributions.exponnorm.fit(DATA.query('CorrectCommission==1&summary_state==1&subid==@sub_name')['ReactionTime'])
    params_exg_State1_all_flip[sub_i,:] = stats.distributions.exponnorm.fit(DATA.query('CorrectCommission==1&summary_state==0&subid==@sub_name')['ReactionTime']*-1)
    params_exg_State2_all_flip[sub_i,:] = stats.distributions.exponnorm.fit(DATA.query('CorrectCommission==1&summary_state==1&subid==@sub_name')['ReactionTime']*-1)


df_skewness = pd.DataFrame({'skewness':skewness.flatten(),'subid':subs})
use_sub = list(df_skewness.query("skewness>0")['subid'])
negative_skew_sub = list(df_skewness.query("skewness<0")['subid'])

df = pd.DataFrame({'mu':params_exg_State1_all[:,1],'sigma':params_exg_State1_all[:,2],'tau':params_exg_State1_all[:,0]*params_exg_State1_all[:,2],'State':"State1",'subid':subs})
df = df.append(pd.DataFrame({'mu':params_exg_State2_all[:,1],'sigma':params_exg_State2_all[:,2],'tau':params_exg_State2_all[:,0]*params_exg_State2_all[:,2],'State':"State2",'subid':subs}))
df = df.query('subid==@use_sub')
df.reset_index(drop=True,inplace=True)

df_flip = pd.DataFrame({'mu':params_exg_State1_all_flip[:,1]*-1,'sigma':params_exg_State1_all_flip[:,2],'tau':params_exg_State1_all_flip[:,0]*params_exg_State1_all_flip[:,2],'State':"State1",'subid':subs})
df_flip = df_flip.append(pd.DataFrame({'mu':params_exg_State2_all_flip[:,1]*-1,'sigma':params_exg_State2_all_flip[:,2],'tau':params_exg_State2_all_flip[:,0]*params_exg_State2_all_flip[:,2],'State':"State2",'subid':subs}))
df_flip = df_flip.query('subid==@negative_skew_sub')
df_flip.reset_index(drop=True,inplace=True)


plt.figure(figsize=(16,10))
# sns.histplot(DATA.query('CorrectCommission==1 and subid==@use_sub'),x = 'ReactionTime',stat="probability",
sns.histplot(DATA.query('CorrectCommission==1 and subid==@use_sub'),x = 'ReactionTime',stat="density",
             common_norm=False,fill=True, hue='summary_state',bins=bins)
if save_flag==1:plt.savefig(fig_dir + 'RT_distribution_betweenBrainStates.png', bbox_inches='tight')
if save_flag==1:plt.savefig(fig_dir + 'RT_distribution_betweenBrainStates.pdf', bbox_inches='tight')

data.plot(kind='hist', bins=bins, density=True, alpha=0.5, label='Data', legend=False, color="#525252")


MyPalette = ["#67a9cf","#ef8a62"]
f = plt.figure(figsize=(16,10))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1,3,1)
ax = sns.barplot(y='mu',x='State', data=df,palette=MyPalette,ci=None)
sns.stripplot(x='State',y='mu',data=df,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],[df.query('State=="State1"').mu,df.query('State=="State2"').mu],color='black',alpha=0.3)
ax.set_ylabel("mu", fontsize=30) #title
ax.set_xlabel("") #title
plt.subplot(1,3,2)
ax = sns.barplot(y='sigma',x='State', data=df,palette=MyPalette,ci=None)
sns.stripplot(x='State',y='sigma',data=df,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],[df.query('State=="State1"').sigma,df.query('State=="State2"').sigma],color='black',alpha=0.3)
ax.set_ylabel("sigma", fontsize=30) #title
ax.set_xlabel("") #title
plt.subplot(1,3,3)
ax = sns.barplot(y='tau',x='State', data=df,palette=MyPalette,ci=None)
sns.stripplot(x='State',y='tau',data=df,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],[df.query('State=="State1"').tau,df.query('State=="State2"').tau],color='black',alpha=0.3)
ax.set_ylabel("tau", fontsize=30) #title
ax.set_xlabel("") #title
if save_flag==1:plt.savefig(fig_dir + 'RT_distribution_betweenBrainStates_Statistics.png', bbox_inches='tight')
if save_flag==1:plt.savefig(fig_dir + 'RT_distribution_betweenBrainStates_Statistics.pdf', bbox_inches='tight')


## statistical analysis
g_mu = CalculateEffect(df.query('State=="State1"').reset_index(drop=True).mu,df.query('State=="State2"').reset_index(drop=True).mu)
print(stats.ttest_rel(df.query('State=="State1"').reset_index().mu,df.query('State=="State2"').reset_index().mu), g_mu)
g_sigma = CalculateEffect(df.query('State=="State1"').reset_index(drop=True).sigma,df.query('State=="State2"').reset_index(drop=True).sigma)
print(stats.ttest_rel(df.query('State=="State1"').reset_index().sigma,df.query('State=="State2"').reset_index().sigma), g_sigma)
g_tau = CalculateEffect(df.query('State=="State1"').reset_index(drop=True).tau,df.query('State=="State2"').reset_index(drop=True).tau)
print(stats.ttest_rel(df.query('State=="State1"').reset_index().tau,df.query('State=="State2"').reset_index().tau),g_tau)

df_all = df.append(df_flip)
f = plt.figure(figsize=(16,10))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1,3,1)
ax = sns.barplot(y='mu',x='State', data=df_all,palette=MyPalette,ci=None)
sns.stripplot(x='State',y='mu',data=df_all,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],[df_all.query('State=="State1"').mu,df_all.query('State=="State2"').mu],color='black',alpha=0.3)
ax.set_ylabel("mu", fontsize=30) #title
ax.set_xlabel("") #title
plt.subplot(1,3,2)
ax = sns.barplot(y='sigma',x='State', data=df_all,palette=MyPalette,ci=None)
sns.stripplot(x='State',y='sigma',data=df_all,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],[df_all.query('State=="State1"').sigma,df_all.query('State=="State2"').sigma],color='black',alpha=0.3)
ax.set_ylabel("sigma", fontsize=30) #title
ax.set_xlabel("") #title
plt.subplot(1,3,3)
ax = sns.barplot(y='tau',x='State', data=df_all,palette=MyPalette,ci=None)
sns.stripplot(x='State',y='tau',data=df_all,jitter=0,color='black',alpha=0.3)
plt.plot([0,1],[df_all.query('State=="State1"').tau,df_all.query('State=="State2"').tau],color='black',alpha=0.3)
ax.set_ylabel("tau", fontsize=30) #title
ax.set_xlabel("") #title
if save_flag==1:plt.savefig(fig_dir + 'RT_distribution_betweenBrainStates_Statistics_include_negativeSkewness.png', bbox_inches='tight')
if save_flag==1:plt.savefig(fig_dir + 'RT_distribution_betweenBrainStates_Statistics_include_negativeSkewness.pdf', bbox_inches='tight')

# df_all.to_csv(out_dir + 'Fig3.csv')

## statistical analysis
g_mu = CalculateEffect(df_all.query('State=="State1"').reset_index(drop=True).mu,df_all.query('State=="State2"').reset_index(drop=True).mu)
print(stats.ttest_rel(df_all.query('State=="State1"').reset_index().mu,df_all.query('State=="State2"').reset_index().mu), g_mu)
g_sigma = CalculateEffect(df_all.query('State=="State1"').reset_index(drop=True).sigma,df_all.query('State=="State2"').reset_index(drop=True).sigma)
print(stats.ttest_rel(df_all.query('State=="State1"').reset_index().sigma,df_all.query('State=="State2"').reset_index().sigma), g_sigma)
g_tau = CalculateEffect(df_all.query('State=="State1"').reset_index(drop=True).tau,df_all.query('State=="State2"').reset_index(drop=True).tau)
print(stats.ttest_rel(df_all.query('State=="State1"').reset_index().tau,df_all.query('State=="State2"').reset_index().tau),g_tau)
