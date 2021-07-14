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
import scipy.stats as st
import scipy.io
import seaborn as sns
import numpy.matlib as mb


def calculate_performance(DATA):
    df = pd.DataFrame()
    ## comparing of VTC
    df['RT'] = DATA.query('CorrectCommission==1').groupby(['subid'])['ReactionTime'].mean()
    df['VC'] = DATA.query('CorrectCommission==1').groupby(['subid'])['ReactionTime'].var()
    df['Skewness'] = DATA.query('CorrectCommission==1').groupby(['subid'])['ReactionTime'].skew()
    params_exg_all = np.zeros([sub_num,3]);aic_exg_all = np.zeros([sub_num,1]);r_squared_exg_all = np.zeros([sub_num,1])
    params_exg_flip_zero_all = np.zeros([sub_num,3]);aic_exg_flip_zero_all = np.zeros([sub_num,1]);r_squared_exg_flip_zero_all = np.zeros([sub_num,1])
    bins=200
    for sub_i in range(sub_num):
        sub_name = 'sub-%s' %str(sub_i).zfill(5)
        print(sub_name)
        data = DATA.query('CorrectCommission==1&subid==@sub_name&ReactionTime>0.24&ReactionTime<1.36')['ReactionTime']
        y, x = np.histogram(data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        params_exg_all[sub_i,:] = stats.distributions.exponnorm.fit(data)
        arg = params_exg_all[sub_i,:-2]
        loc = params_exg_all[sub_i,-2]
        scale = params_exg_all[sub_i,-1]
        r_squared_exg_all[sub_i] = np.corrcoef(np.sort(data),np.sort(scale*(st.exponnorm.rvs(*arg,size=len(data)))+loc))[0,1]**2
        logLik = -np.sum( st.exponnorm.logpdf(x, loc=loc, scale=scale, *arg)) 
        aic_exg_all[sub_i] = 2*3-2*logLik

        params_exg_flip_zero_all[sub_i,:] = stats.distributions.exponnorm.fit(-data)
        arg = params_exg_flip_zero_all[sub_i,:-2]
        loc = params_exg_flip_zero_all[sub_i,-2]
        scale = params_exg_flip_zero_all[sub_i,-1]
        r_squared_exg_flip_zero_all[sub_i] = np.corrcoef(np.sort(-data),np.sort(scale*(st.exponnorm.rvs(*arg,size=len(-data)))+loc))[0,1]**2
        y, x = np.histogram(-data, bins=bins, density=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0
        logLik = -np.sum( st.exponnorm.logpdf(x, loc=loc, scale=scale, *arg)) 
        aic_exg_flip_zero_all[sub_i] = 2*3-2*logLik

        

    df['mu'] = params_exg_all[:,1]
    df['sigma'] = params_exg_all[:,2]
    df['tau'] = params_exg_all[:,0]*params_exg_all[:,2]
    df['aic_exg'] = aic_exg_all 
    df['r_square_exg'] = r_squared_exg_all

    df['mu_flip_zero'] = params_exg_flip_zero_all[:,1]
    df['sigma_flip_zero'] = params_exg_flip_zero_all[:,2]
    df['tau_flip_zero'] = params_exg_flip_zero_all[:,0]*params_exg_flip_zero_all[:,2]
    df['aic_exg_flip_zero'] = aic_exg_flip_zero_all
    df['r_square_exg_flip_zero'] = r_squared_exg_flip_zero_all

    return(df)

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


# df_1st_half = calculate_performance(DATA.query('trial_timing=="1st"|trial_timing=="2nd"'))
# df_2nd_half = calculate_performance(DATA.query('trial_timing=="3rd"|trial_timing=="4th"'))

# df_1st_half['mu_summary'] = np.where(df_1st_half.Skewness>0,df_1st_half.mu,df_1st_half.mu_flip_zero*-1)
# df_1st_half['sigma_summary'] = np.where(df_1st_half.Skewness>0,df_1st_half.sigma,df_1st_half.sigma_flip_zero)
# df_1st_half['tau_summary'] = np.where(df_1st_half.Skewness>0,df_1st_half.tau,df_1st_half.tau_flip_zero)
# df_1st_half['group'] = np.where(df_1st_half.Skewness>0,"Positive","Negative")
# df_1st_half.to_csv(out_dir + 'exGaussian_1stHalf.csv')

# df_2nd_half['mu_summary'] = np.where(df_2nd_half.Skewness>0,df_2nd_half.mu,df_2nd_half.mu_flip_zero*-1)
# df_2nd_half['sigma_summary'] = np.where(df_2nd_half.Skewness>0,df_2nd_half.sigma,df_2nd_half.sigma_flip_zero)
# df_2nd_half['tau_summary'] = np.where(df_2nd_half.Skewness>0,df_2nd_half.tau,df_2nd_half.tau_flip_zero)
# df_2nd_half['group'] = np.where(df_2nd_half.Skewness>0,"Positive","Negative")
# df_2nd_half.to_csv(out_dir + 'exGaussian_2ndHalf.csv')

df_1st_half = pd.read_csv(out_dir + 'exGaussian_1stHalf.csv')
df_2nd_half = pd.read_csv(out_dir + 'exGaussian_2ndHalf.csv')


# for i in ['mu_summary','sigma_summary','tau_summary','mu','sigma','tau','mu_flip_zero','sigma_flip_zero','tau_flip_zero']:
for i in ['mu_summary','sigma_summary','tau_summary']:
    r,p =stats.spearmanr(df_1st_half[i], df_2nd_half[i])
    Reliability = (2*r)/(1+r)
    print(i,", Reliability = %s" %Reliability)


# df = calculate_performance(DATA)
# df['mu_summary'] = np.where(df.Skewness>0,df.mu,df.mu_flip_zero*-1)
# df['sigma_summary'] = np.where(df.Skewness>0,df.sigma,df.sigma_flip_zero)
# df['tau_summary'] = np.where(df.Skewness>0,df.tau,df.tau_flip_zero)
# df['group'] = np.where(df.Skewness>0,"Positive","Negative")
# df.to_csv(out_dir + 'exGaussian.csv')

df = pd.read_csv(out_dir + 'exGaussian.csv')

print('mu: mean = %s, std = %s, skewness = %s, kurtosis = %s' %(round(df.query("group=='Positive'").mu_summary.mean(),2),round(df.query("group=='Positive'").mu_summary.std(),2),
                                                                round(df.query("group=='Positive'").mu_summary.skew(),2),round(df.query("group=='Positive'").mu_summary.kurt(),2)))
print('sigma: mean = %s, std = %s, skewness = %s, kurtosis = %s'  %(round(df.query("group=='Positive'").sigma_summary.mean(),2),round(df.query("group=='Positive'").sigma_summary.std(),2),
                                                                round(df.query("group=='Positive'").sigma_summary.skew(),2),round(df.query("group=='Positive'").sigma_summary.kurt(),2)))
print('tau: mean = %s, std = %s, skewness = %s, kurtosis = %s' %(round(df.query("group=='Positive'").tau_summary.mean(),2),round(df.query("group=='Positive'").tau_summary.std(),2),
                                                                round(df.query("group=='Positive'").tau_summary.skew(),2),round(df.query("group=='Positive'").tau_summary.kurt(),2)))

print('mu: mean = %s, std = %s, skewness = %s, kurtosis = %s' %(round(df.query("group=='Negative'").mu_summary.mean(),2),round(df.query("group=='Negative'").mu_summary.std(),2),
                                                                round(df.query("group=='Negative'").mu_summary.skew(),2),round(df.query("group=='Negative'").mu_summary.kurt(),2)))
print('sigma: mean = %s, std = %s, skewness = %s, kurtosis = %s'  %(round(df.query("group=='Negative'").sigma_summary.mean(),2),round(df.query("group=='Negative'").sigma_summary.std(),2),
                                                                round(df.query("group=='Negative'").sigma_summary.skew(),2),round(df.query("group=='Negative'").sigma_summary.kurt(),2)))
print('tau: mean = %s, std = %s, skewness = %s, kurtosis = %s' %(round(df.query("group=='Negative'").tau_summary.mean(),2),round(df.query("group=='Negative'").tau_summary.std(),2),
                                                                round(df.query("group=='Negative'").tau_summary.skew(),2),round(df.query("group=='Negative'").tau_summary.kurt(),2)))


print('mu: mean = %s, std = %s, skewness = %s, kurtosis = %s' %(round(df.mu_summary.mean(),2),round(df.mu_summary.std(),2),
                                                                round(df.mu_summary.skew(),2),round(df.mu_summary.kurt(),2)))
print('sigma: mean = %s, std = %s, skewness = %s, kurtosis = %s'  %(round(df.sigma_summary.mean(),2),round(df.sigma_summary.std(),2),
                                                                round(df.sigma_summary.skew(),2),round(df.sigma_summary.kurt(),2)))
print('tau: mean = %s, std = %s, skewness = %s, kurtosis = %s' %(round(df.tau_summary.mean(),2),round(df.tau_summary.std(),2),
                                                                round(df.tau_summary.skew(),2),round(df.tau_summary.kurt(),2)))