#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 11:23:18 2019

@author: seunglee
"""

from ethnicolr import census_ln
import pandas as pd
from os import listdir
from os.path import isfile, join

root = '/Users/seunglee/Downloads/project/'

# r=root, d=directories, f = files

df_file2 = pd.DataFrame()

counter = 0
stop = 0
name_found = 0
for r, d, f in os.walk(path):
    if('cleaned' in r):
        continue
    if(stop>0):
        break
    if(('census' in r)|('collegiate_times' in r)):
        continue
    for file in f:
        if('df' in file):
            continue
        if(stop>0):
            break
        if (('.csv' in file)|('.7z' in file)):
            if('.7z' not in file):                
                temp = pd.read_csv(os.path.join(r, file), encoding='latin-1', error_bad_lines=False)
                if(pd.isna(temp.index).sum()>100):
                    temp_cols = temp.columns[1:]
                    del temp[temp.columns[-1]]
                    temp.columns = temp_cols
                columns = pd.DataFrame(temp.dtypes)
                columns['r'] = r
                columns['file'] = file
                
                df_file2.loc[counter, 'file'] = file
                df_file2.loc[counter, 'r'] = r
                counter_name = 0
                counter_salary = 0
                
                for col in columns.index:
                    if(col=='Unnamed: 0'):
                        continue
                    col_lower = col.lower()
                    if(('name' in col_lower)|('employee' in col_lower)|('payee' in col_lower)):
                        if(temp[col].dtype=='O'):
                            temp[col].fillna('', inplace=True)
                            try:
                                if(('first' in col_lower)|('fname' in col_lower)):
                                    temp['first_name'] = temp[col]
                                elif(('last' in col_lower)|('lname' in col_lower)):
                                    temp['last_name'] = temp[col]
                                elif(temp[col].str.contains(',').sum()>temp.shape[0]/2):
                                    temp['first_name'] = temp[col].str.split(',').str[1].str.strip().str.split(' ').str[0]
                                    temp['last_name'] = temp[col].str.split(',').str[0].str.strip()
                                else:
                                    temp['first_name'] = temp[col].str.split(' ').str[0].str.strip()
                                    temp['last_name'] = temp[col].str.split(' ').str[1].str.strip()    
                                    temp.loc[temp.last_name.str.contains('.')] = temp[col].str.split('.').str[2].str.strip() 
                                name_found = 1
                                df_file2.loc[counter, 'name_'+str(counter_name)] = col_lower
                                counter_name+=1
                                
                            except:
                                print(r, file, col, 'not name')
                    if(('salary' in col_lower)|('compensation' in col_lower)|
                            ('wage' in col_lower)|('total' in col_lower)|
                            ('earnings' in col_lower)|('overtime' in col_lower)):
                        if(temp[col].dtype=='O'):
                            temp[col] = temp[col].str.replace(',', '')
                            temp[col] = temp[col].str.replace('$', '')
                            try:
                                temp[col] = temp[col].astype('double')
                            except:
                                print(r, file, col, 'not double')
                        else:
                            temp[col] = temp[col].astype('double')
                        df_file2.loc[counter, 'salary_'+str(counter_salary)] = col_lower
                        counter_salary+=1
                                
                temp.to_csv('/Users/seunglee/Downloads/public_salaries-master/cleaned/df_'+str(counter)+'.csv')
                counter+=1
                stop=0
                print(counter)
                




temp = pd.read_csv('/Users/seunglee/Downloads/public_salaries-master/ia/state_2006_2016.csv')

col = 'Employee'
temp['first_name'] = ''
temp.loc[temp[col].str.contains(','), 'first_name'] = temp[col].str.split(',').str[1].str.strip().str.split(' ').str[0]
temp.loc[~temp[col].str.contains(','), 'first_name'] = temp[col].str.split(' ').str[1].str.strip().str.split(' ').str[0]

temp['last_name'] = ''
temp.loc[temp[col].str.count(' ')==1, 'last_name'] = temp[col].str.split(' ').str[1].str.strip()
temp.loc[temp[col].str.count(' ')==2, 'last_name'] = temp[col].str.split(' ').str[2].str.strip()
a = temp.head(1000)

temp['count'] = temp[col].str.count(" ")
temp = temp[[']]
df_file3 = df_file2.loc[~(pd.isna(df_file2.name_0))&~(pd.isna(df_file2.salary_0))]


temp['first_name'] = temp['Name']
temp['last_name'] = temp['Name.1']


###################################################
#
# Iterated over the cleaned by files for full time wage and keep columns for all the ones cleaned above
# Manual iterate
############################
n = 32
file_old = '/Users/seunglee/Downloads/public_salaries-master/cleaned/df_'+str(n)+'.csv'
temp = pd.read_csv(file_old)
a = temp.head(1000)
#
#temp['first_name'] = temp[col].str.split(',').str[1].str.strip().str.split(' ').str[0]
#temp['last_name'] = temp[col].str.split(',').str[0].str.strip()


cols = ['first_name', 'last_name', 'EARNINGS', 'file']
temp['file'] = df_file3.loc[n, 'r'] + df_file3.loc[n, 'file']

temp = temp.loc[temp['ACCOUNT_DESCRIPTION']=='Salaries & Wages-Full Time']
file_new = '/Users/seunglee/Downloads/public_salaries-master/cleaned2/df_'+str(n)+'.csv'

temp.loc[:, cols].to_csv(file_new, index=False)

for year in ['2009', '2010', '2011', '2012', '2013']:
    temp2 = temp.loc[temp['Year']==int(year)]
    file_new = '/Users/seunglee/Downloads/public_salaries-master/cleaned2/df_'+str(n)+'_'+year+'.csv'
    temp2.loc[:, cols].to_csv(file_new, index=False)
    
temp.loc[:, cols].to_csv(file_new, index=False)
   
###########################################################################    
#    
# Run the race impute models
#
######################
df_description = pd.DataFrame()

counter=0
path = '/Users/seunglee/Downloads/public_salaries-master/cleaned2/'
for r, d, f in os.walk(path):
    for file in f:
        counter+=1
        if(counter<11):
            continue
        counter+=1
        if('.csv' not in file):
            continue
        df = pd.read_csv(r+'/'+file)
        original_columns = list(df.columns)
        models_str = []
        for model in [census_ln, pred_census_ln, pred_wiki_ln, pred_wiki_name, pred_fl_reg_ln, pred_fl_reg_name]:
            model_name = model.__name__
            models_str.append('race_'+model_name)
            if(model_name=='census_ln'): 
                df = model(df, 'last_name')
                df.replace('(S)', '0', inplace=True)
                race_cols = df.columns[df.columns.str.startswith('pct')]
                for col in race_cols:
                    df[col] = df[col].astype('double')
                df['race'] = df[race_cols].idxmax(axis=1)
            elif(model_name=='pred_wiki_name'):
                df = model(df, 'first_name', 'last_name')
            elif(model_name=='pred_fl_reg_name'):
                df = model(df, lname_col='last_name', fname_col='first_name')
            else:
                df = model(df, 'last_name')
            df.rename(columns={'race':'race_'+model_name}, inplace=True) 
        df = df[original_columns+models_str]
        df.to_csv(path+'modelran_'+file)
        print('done:', file)                

#########################################################################
#
# Gb summaries of race by file and year
#
############



import re
counter=0
path = '/Users/seunglee/Downloads/public_salaries-master/cleaned2/'
for r, d, f in os.walk(path):
    for file in f:
#        counter+=1
#        if(counter<11):
#            continue
        if('modelran' not in file):
            continue
        fname = r+file
        counter+=1
        print(counter, fname)
        temp = pd.read_csv(r+file)
        if(temp[temp.columns[3]].dtype=='O'):              
            if(any(temp[temp.columns[3]].str.contains('\('))):
                temp = temp.loc[~temp[temp.columns[3]].str.contains('\(')]
            if(any(temp[temp.columns[3]].str.contains('$'))):
                temp[temp.columns[3]] = temp[temp.columns[3]].str.replace('$', '')
            if(any(temp[temp.columns[3]].str.contains(','))):
                temp[temp.columns[3]] = temp[temp.columns[3]].str.replace(',', '')                
            temp[temp.columns[3]] = temp[temp.columns[3]].astype('float')
        temp_str = temp['file'][0]
        year_0 = re.findall('[0-9]{4}', fname)
        year_1 = re.findall('[0-9]{4}', temp_str)
        
        state = temp_str.split('salaries-master/')[1][:2]
        
        if(len(year_0)>0):
            year = year_0[0]
        elif(len(year_1)>0):
            year = year_1[0]
        else:
            year = 'noyear'
        
        paycol = temp.columns[3]
        temp = temp.loc[temp[paycol]>0]
        n = temp.shape[0]
        gb_summary = pd.DataFrame(index=['white', 'hispanic', 'black', 'asian', 'all'])
        gb_summary['state'] = state
        gb_summary['year'] = year
        for model in ['race_census_ln',	'race_pred_census_ln',
                      'race_pred_fl_reg_ln',	 'race_pred_fl_reg_name']:
            
            gb_summary0 = temp.groupby(model).mean()[[paycol]]
            gb_summary1 = temp.groupby(model).count()[[paycol]]/n

            gb_summary0.columns = [model+'_salary']
            gb_summary1.columns = [model+'_composition']

            gb_summary0 = gb_summary0.merge(gb_summary1, how='inner', left_index=True, right_index=True)
            for ind in gb_summary0.index:
                if('white' in ind):
                    gb_summary0.loc[ind, 'race'] = 'white'
                elif('hispanic' in ind):
                    gb_summary0.loc[ind, 'race'] = 'hispanic'
                elif('black' in ind):
                    gb_summary0.loc[ind, 'race'] = 'black'
                elif(('asian' in ind)|('api' in ind)):
                    gb_summary0.loc[ind, 'race'] = 'asian' 
                    
            gb_summary0 = gb_summary0.loc[~pd.isna(gb_summary0.race)] 
            gb_summary0.loc['all', model+'_salary'] = temp[paycol].mean()     
            gb_summary0.loc['all', model+'_composition'] = temp.shape[0]  
            gb_summary0.loc['all', 'race'] = 'all'
            gb_summary0.set_index('race', inplace=True)
            gb_summary = gb_summary.merge(gb_summary0, how='inner', left_index=True, right_index=True)
            
        temp_str = fname.replace('modelran', 'groupby')
        gb_summary.to_csv(temp_str)
        
        
        