# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:33:06 2020

@author: ayumu
"""

import numpy as np
import glob
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
from collections import Counter
import warnings
import scipy.stats as st
import statsmodels as sm
import matplotlib
import sys

def best_fit_distribution(data,DISTRIBUTIONS, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    model = []
    color_map = ["#66c2a5","#fc8d62","#8da0cb","#e78ac3"]
    for i, distribution in enumerate(DISTRIBUTIONS):
            params = distribution.fit(data)
            color = color_map[i]
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]
            pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
            rss = np.sum(np.power(y - pdf, 2.0)) #residual sum of squares = rss
            tss = np.sum((y-np.mean(y))**2) #total sum of squares = tss
            # r_squared = 1- (rss/tss)
            r_squared = np.corrcoef(np.sort(data),np.sort(scale*(distribution.rvs(*arg,size=len(data),random_state=55))+loc))[0,1]**2
            logLik = -np.sum( distribution.logpdf(x, loc=loc, scale=scale, *arg)) 
            k = len(params)
            AIC = 2*k-2*logLik
            model.append([
                distribution.name,
                params,
                r_squared,
                bins,
                color,
                AIC,
                ])
    return model

# matplotlib.style.use('ggplot')

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

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    start = dist.ppf(0.001, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.001, loc=loc, scale=scale)
    end = dist.ppf(0.999, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.999, loc=loc, scale=scale)

    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)
    return pdf


def plot(data,i,reward):
    fit_name, fit_params, fit_r_squared,bins,color = fit_list[i]
    dist = getattr(st, fit_name)
    # Make PDF with best params 
    pdf = make_pdf(dist, fit_params)

    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=4, label='PDF', legend=True,color=color)
    data.plot(kind='hist', bins=bins, density=True, alpha=0.5, label='Data', legend=True, ax=ax,color="#525252")

    param_names = (dist.shapes + ', loc, scale').split(', ') if dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, fit_params)])
    dist_str = '{}({}) r_squared={}'.format(fit_name, param_str, fit_r_squared)

    ax.set_title(dist_str)
    ax.set_xlabel(u'Reaction time (s)')
    ax.set_ylabel('Frequency')
    plt.show()
    # plt.savefig(fig_dir + 'RT_distribution_fitting_by_%s%s' %(fit_name,reward))

##############
# parameters #
##############
    
save_flag = 1


# project = 'OriginalGradCPT'
project = 'GradCPT_MindWandering'
# project = 'GradCPT_reward'

top_dir = 'C:/Users/ayumu/Dropbox/gradCPT/'
source_dir = 'C:/Users/ayumu/Dropbox/gradCPT/data/%s/' %project
fig_dir = top_dir + 'fig/%s/behavior/' %project
if os.path.isdir(fig_dir)==False: os.mkdir(fig_dir)

out_dir = source_dir + '/behavior/'
if os.path.isdir(out_dir)==False: os.mkdir(out_dir)

if project == 'GradCPT_MindWandering':
    tr=1.08
    isi=1.3
    bins = 20
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants_HC.tsv')[0],delimiter='\t')
#     demo = pd.read_csv(glob.glob(top_dir + '/code/participants_ADHD.tsv')[0],delimiter='\t')
    # demo = pd.read_csv(glob.glob(top_dir + '/code/participants_MW.tsv')[0],delimiter='\t')
    reward = ''
    ylim = [0.92, 1]
    fontsize = 7
elif project == 'GradCPT_reward':
    tr=2
    isi=0.8
    bins = 20
    target = 'IndividualSubject'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants_reward.tsv')[0],delimiter='\t')
    # reward = '_Reward'
    # reward = '_Nonreward'
    reward = ''
    ylim = [0.85, 1]
    fontsize = 12
elif project == 'OriginalGradCPT':
    tr=2
    isi=0.8
    bins = 15
    target = 'IndividualSubject'
    demo = pd.read_csv(glob.glob(top_dir + '/code/participants.tsv')[0],delimiter='\t')
    reward = ''
    ylim = [0.92, 1]
    fontsize = 12

subs = demo['participants_id']
sub_num = len(subs)

# extract signals
df = pd.DataFrame()
for sub_i in subs:
    dprime = []
    criterion = []
    trial_num = []
    CE = []
    OE = []
    max_no_response_time = []
    if project == 'GradCPT_MindWandering':
        task_file = glob.glob(source_dir + '/MRI/'+ sub_i +'_task-gradCPTMW*events.tsv');task_file.sort()
    else: 
        task_file = glob.glob(source_dir + '/MRI/'+ sub_i +'_task-gradCPT*events.tsv');task_file.sort()
    DATA_behave = pd.DataFrame()
    for num_file_i,task_file_i in enumerate(task_file,1):
        taskinfo = pd.read_csv(task_file_i,delimiter='\t')
        taskinfo['trial'] = num_file_i      
        DATA_behave = DATA_behave.append(taskinfo)
    DATA_behave['subid'] = sub_i
    df = df.append(DATA_behave)

sns.set_context('poster')

DISTRIBUTIONS = [        
   st.exponnorm,st.norm
]

df_R_squared = pd.DataFrame()
df_AIC = pd.DataFrame()
df_behavior = pd.DataFrame()
for sub_i in subs:
    df_sub = df.query('subid == @sub_i').reset_index(drop=True)
    pre_trial = np.where(df_sub.CommissionError+df_sub.OmissionError)[0]-1
    post_trial = np.where(df_sub.CommissionError+df_sub.OmissionError)[0]+1
    if pre_trial[0]==-1:
       pre_trial = pre_trial[1:]
       post_trial = post_trial[1:]
    
    if reward == '_Reward':
        data = df.query('CorrectCommission==1 & Reward_trial==1 & subid == @sub_i')['ReactionTime']
    elif reward == '_Nonreward':
        data = df.query('CorrectCommission==1 & Reward_trial==0 & subid == @sub_i')['ReactionTime']
    else:
        data = df.query('CorrectCommission==1 & subid == @sub_i')['ReactionTime']
    fit_list = best_fit_distribution(data,DISTRIBUTIONS,bins=bins, ax=None)
    # fit_list.sort(key=lambda x: x[2],reverse=True)
    R_squared = pd.DataFrame()
    AIC = pd.DataFrame()
    new_color_map = []
    for i, model_i in enumerate(DISTRIBUTIONS):
        R_squared[fit_list[i][0]] = [fit_list[i][2]]
        AIC[fit_list[i][0]] = [fit_list[i][5]]
    R_squared = R_squared.T.reset_index()
    R_squared['subid'] = sub_i
    df_R_squared = df_R_squared.append(R_squared)
    AIC = AIC.T.reset_index()
    AIC['subid'] = sub_i
    df_AIC = df_AIC.append(AIC)
    K, mu, sigma = fit_list[0][1]
    mu_norm, sigma_norm = fit_list[1][1]
    tau = K*sigma
    skew = data.skew()
    pes = np.mean(df_sub.T[post_trial].T.query('CorrectCommission==1')['ReactionTime'])-np.mean(df_sub.T[pre_trial].T.query('CorrectCommission==1')['ReactionTime'])
    df_behavior_sub = pd.DataFrame({'PES':[pes],'mu':[mu],'sigma':[sigma],'skew':[skew],'tau':[tau],'mu_norm':[mu_norm],'sigma_norm':[sigma_norm],'subid':sub_i})
    df_behavior = df_behavior.append(df_behavior_sub)

not_use_sub = list(df_behavior.query('skew<0')['subid'])

df_behavior.reset_index(drop=True,inplace=True)
df_R_squared.rename(columns={0:'R_squared'},inplace=True)
df_R_squared.reset_index(inplace=True)
df_AIC.rename(columns={0:'AIC'},inplace=True)
df_AIC.reset_index(inplace=True)
color_map = ["#66c2a5","#fc8d62","#8da0cb","#e78ac3"]
order=['exponnorm','norm']
print(df_behavior.tau)

new_sub_num = sub_num-len(not_use_sub)

plt.figure(figsize=(5,5))
sns.barplot(x="index",y='R_squared',data=df_R_squared.query('subid!=@not_use_sub'),
            order=order,palette=color_map,ci=None)
sns.stripplot(x='index',y='R_squared',data=df_R_squared.query('subid!=@not_use_sub'),jitter=0,order=order,color='black',alpha=0.3,linewidth=0.5)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
plt.plot(range(len(DISTRIBUTIONS)),np.reshape(np.array(df_R_squared.query('subid!=@not_use_sub')['R_squared']),[new_sub_num,len(DISTRIBUTIONS)]).T,color='black',alpha=0.3)
plt.xlabel('')
plt.ylabel('R squared')
plt.ylim(ylim)
if save_flag==1:plt.savefig(fig_dir + 'R_squared_eachSubject%s.pdf' %reward, bbox_inches='tight')
if save_flag==1:plt.savefig(fig_dir + 'R_squared_eachSubject%s' %reward, bbox_inches='tight')

g = CalculateEffect(df_R_squared.query('index=="exponnorm"')['R_squared'],df_R_squared.query('index=="norm"')['R_squared'])
print(st.ttest_rel(df_R_squared.query('index=="exponnorm"')['R_squared'],df_R_squared.query('index=="norm"')['R_squared']),g)

df_R_squared.to_csv(out_dir + 'Fig4_1.csv')


# plt.figure(figsize=(8,5))
# sns.barplot(x="index",y='AIC',data=df_AIC,
#             order=order,palette=color_map,ci=None)
# sns.stripplot(x='index',y='AIC',data=df_AIC,jitter=0,order=order,hue='subid',palette="hls",linewidth=0.5)
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=fontsize)
# plt.plot(range(len(DISTRIBUTIONS)),np.reshape(np.array(df_AIC['AIC']),[sub_num,len(DISTRIBUTIONS)]).T,color='black',alpha=0.3)
# plt.xlabel('')
# plt.ylabel('AIC')
# # plt.ylim(ylim)
# plt.savefig(fig_dir + 'AIC_eachSubject%s' %reward, bbox_inches='tight')