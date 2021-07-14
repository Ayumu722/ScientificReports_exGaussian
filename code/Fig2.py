# #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017 1016

@author: ayumu
"""

#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
from scipy import stats
import scipy.io
import numpy.matlib as mb
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

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
age = mat['stats'][:,4]
sub_num = len(mat['cleanrt'])
df = pd.read_csv(out_dir + 'participants.csv')
DATA = pd.DataFrame()
DATA['trial_type'] = mb.repmat(np.where(Trial==1,'mountain','city'),1,sub_num)[0]
DATA['trial_type'] = mb.repmat(np.where(Trial==1,'mountain','city'),1,sub_num)[0]
DATA['response'] = mat['cleanresponse'].flatten()
DATA['CorrectOmission'] = np.where((DATA['response']==1)&(DATA['trial_type']=='mountain'),1,0)
DATA['CorrectCommission'] = np.where((DATA['response']==1)&(DATA['trial_type']=='city'),1,0)
DATA['CommissionError'] = np.where(DATA['response']==-1,1,0)
DATA['OmissionError'] = np.where(DATA['response']==0,1,0)
DATA['ReactionTime'] =mat['cleanrt'].flatten()/1000
subs = ['sub-%s'%str(sub_i).zfill(5) for sub_i in range(sub_num)]
DATA['subid'] = mb.repmat(subs,300,1).T.flatten()

df['group'] = np.where(df.Skewness>0,'Over','Under')
df['age'] = age

g = CalculateEffect(df.query('Skewness<0')['dprime'],df.query('Skewness>0')['dprime'])
print(stats.ttest_ind(df.query('Skewness<0')['dprime'],df.query('Skewness>0')['dprime']),g)


g = CalculateEffect(df.query('Skewness<0')['RT'],df.query('Skewness>0')['RT'])
print(stats.ttest_ind(df.query('Skewness<0')['RT'],df.query('Skewness>0')['RT']),g)

sns.set_context('poster')
## exclude subjects
use_sub = list(df.query("a<5 & a>-5")['subid'])
notuse_sub_over = list(df.query("a>5")['subid'])
notuse_sub_under = list(df.query("a<-5")['subid'])


color_map = ["#66c2a5","#fc8d62","#8da0cb","#e78ac3"]
order=['exponnorm','norm']

data = df[['r_square_exg','r_square_norm']].melt()
data.columns = ["type","R_squared"]
data.type = np.where(data.type=="r_square_exg","exponnorm","norm")

plt.figure(figsize=(5,5))
sns.violinplot(x="type", y="R_squared",palette=color_map,data=data)
plt.xlabel('')
plt.ylabel('R squared')

g = CalculateEffect(df.query('Skewness>0')['r_square_exg'],df.query('Skewness>0')['r_square_norm'])
print(stats.ttest_rel(df.query('Skewness>0')['r_square_exg'],df.query('Skewness>0')['r_square_norm']),g)

g = CalculateEffect(df.query('Skewness<0')['r_square_exg_flip_zero'],df.query('Skewness<0')['r_square_norm'])
print(stats.ttest_rel(df.query('Skewness<0')['r_square_exg_flip_zero'],df.query('Skewness<0')['r_square_norm']),g)

print(stats.ttest_rel(data.query('type=="exponnorm"')['R_squared'],data.query('type=="norm"')['R_squared']))
      

df['mu_summary'] = np.where(df.Skewness>0,df.mu,df.mu_flip_zero*-1)
df['sigma_summary'] = np.where(df.Skewness>0,df.sigma,df.sigma_flip_zero)
df['tau_summary'] = np.where(df.Skewness>0,df.tau,df.tau_flip_zero)
df['group'] = np.where(df.Skewness>0,"Positive","Negative")

for j in ['mu_summary','sigma_summary','tau_summary']:
    for i in ['FA','MISS']:
        print(i," and ", j,", correlation = %s, p value = %s" %(stats.spearmanr(df[i], df[j])))
        print(i," and ", j,", correlation = %s, p value = %s, Positive" %(stats.spearmanr(df.query("group=='Positive'")[i], df.query("group=='Positive'")[j])))
        print(i," and ", j,", correlation = %s, p value = %s, Negative" %(stats.spearmanr(df.query("group=='Negative'")[i], df.query("group=='Negative'")[j])))
        plt.figure(figsize=(6, 6))
        g = sns.jointplot(x=j, y=i, kind="scatter",hue="group",data=df, s=3, space=0)
        g.plot_joint(sns.kdeplot, zorder=0, levels=6)
        plt.legend('')
        if save_flag==1:plt.savefig(fig_dir + 'RelationshipBetween_%s_%s_use_both_subjects.pdf' %(i,j), bbox_inches='tight')
        if save_flag==1:plt.savefig(fig_dir + 'RelationshipBetween_%s_%s_use_both_subjects.png' %(i,j), bbox_inches='tight')

data_over = df.query('Skewness>0')[['r_square_exg','r_square_exg_flip_zero','r_square_norm']].melt()
data_over.columns = ["type","R_squared"]
data_over.type = np.where(data_over.type=="r_square_exg","exponnorm",data_over.type)
data_over.type = np.where(data_over.type=="r_square_exg_flip_zero","flip",data_over.type)
data_over.type = np.where(data_over.type=="r_square_norm","norm",data_over.type)
plt.figure(figsize=(5,5))
sns.violinplot(x="type", y="R_squared",palette=color_map,data=data_over)
plt.xlabel('')
plt.ylabel('R squared')
plt.ylim([0.9, 1])
if save_flag==1:plt.savefig(fig_dir + 'Comparison_Fitting_with_flip_positive_skew.png', bbox_inches='tight')
if save_flag==1:plt.savefig(fig_dir + 'Comparison_Fitting_with_flip_positive_skew.pdf', bbox_inches='tight')
print(stats.ttest_rel(data_over.query('type=="exponnorm"')['R_squared'],data_over.query('type=="flip"')['R_squared']))

data_under = df.query('Skewness<0')[['r_square_exg','r_square_exg_flip_zero','r_square_norm']].melt()
data_under.columns = ["type","R_squared"]
data_under.type = np.where(data_under.type=="r_square_exg","exponnorm",data_under.type)
data_under.type = np.where(data_under.type=="r_square_exg_flip_zero","flip",data_under.type)
data_under.type = np.where(data_under.type=="r_square_norm","norm",data_under.type)
plt.figure(figsize=(5,5))
sns.violinplot(x="type", y="R_squared",palette=color_map,data=data_under)
plt.xlabel('')
plt.ylabel('R squared')
plt.ylim([0.9, 1])
if save_flag==1:plt.savefig(fig_dir + 'Comparison_Fitting_with_flip_negative_skew.png', bbox_inches='tight')
if save_flag==1:plt.savefig(fig_dir + 'Comparison_Fitting_with_flip_negative_skew.pdf', bbox_inches='tight')
print(stats.ttest_rel(data_under.query('type=="exponnorm"')['R_squared'],data_under.query('type=="flip"')['R_squared']))


from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

df['tau_summary_standarized'] = stats.zscore(df.tau_summary)
df['mu_summary_standarized'] = stats.zscore(df.mu_summary)
df['sigma_summary_standarized'] = stats.zscore(df.sigma_summary)

## Omission error
model = smf.ols("FA ~ mu_summary_standarized  + sigma_summary_standarized + tau_summary_standarized+ 1", df).fit().summary()
print(model)
## Commission error
model = smf.ols("MISS ~ mu_summary_standarized + sigma_summary_standarized + tau_summary_standarized + 1", df).fit().summary()
print(model)



