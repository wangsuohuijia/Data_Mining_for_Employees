# -*- coding: UTF-8 -*-

# Author: Suohuijia Wang
# ID:     17200170
# Date:   28/04/2018
# Data Mining on the employee dataset


# %matplotlib inline
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pprint import pprint

# part 1
raw_data = pd.read_csv('Hackathon_Data_2018_Rev.csv')
raw_data.head(10)
raw_data.tail(10)
pprint(raw_data.columns.tolist())
describe = raw_data.describe().transpose()
describe.to_excel('describe.xlsx')

# part 2
print(raw_data.shape)
raw_data.info()
print(raw_data.shape)
# step 1: Feature Engineering
## one hot function
def one_hot(df, column, seperate_col=True, discete_dict=None, rename_dict=None):
    df = df.copy()
    if seperate_col:
        one_hot_list = df[column].unique()
        for key in one_hot_list:
            df[key] = 0
            df.loc[df[column] == key, key] = 1
    else:
        one_hot_list = df[column].unique()
        temp = df[column].copy()
        if discete_dict is None:
            i = 0
            for key in sorted(one_hot_list):
                df.loc[temp == key, column] = i
                i += 1
        else:
            for key in discete_dict:
                df.loc[temp == key, column] = discete_dict[key]

    if rename_dict is not None:
        df = df.rename(columns=rename_dict)

    return df

# # one_hot function test
# raw_data_temp = one_hot(raw_data, column='CurrentEmployee',seperate_col=False, discete_dict={'Yes':1, 'No':2})

data = raw_data.copy()

# Current employee
raw_data['CurrentEmployee'].unique()
data = one_hot(data, column='CurrentEmployee',seperate_col=False, discete_dict={'Yes':1, 'No':0})
pd.DataFrame(pd.concat([raw_data['CurrentEmployee'], data['CurrentEmployee']], axis=1))
data['CurrentEmployee'] = data['CurrentEmployee'].astype(np.int64)


# Business travel
raw_data['BusinessTravel'].unique()
data = one_hot(data, column='BusinessTravel')
pd.DataFrame(pd.concat([raw_data['BusinessTravel'], data[['Travel_Rarely','Travel_Frequently','Non-Travel']]], axis=1))
data.drop('BusinessTravel', axis=1, inplace=True)

# Department
raw_data['Department'].unique()
data = one_hot(data, column='Department')
temp = pd.DataFrame(pd.concat([raw_data['Department'], data[['Sales','Research & Development','Human Resources']]], axis=1))
# print(temp)

print('Sales Employee Number is', temp['Sales'].sum())
print('Research & Development Employee Number is', temp['Research & Development'].sum())
print('Human Resources Employee Number is', temp['Human Resources'].sum())
data.drop('Department', axis=1, inplace=True)

# Job role
pprint(raw_data['JobRole'].unique().tolist())

# Sales
raw_data.loc[raw_data['Department'] == 'Sales', 'JobRole'].unique()
print('Sales Executive Avg Salary', raw_data.loc[(raw_data['Department'] == 'Sales') & (raw_data['JobRole'] == 'Sales Executive'), 'MonthlyIncome'].mean())
print('Manager Avg Salary', raw_data.loc[(raw_data['Department'] == 'Sales') & (raw_data['JobRole'] == 'Manager'), 'MonthlyIncome'].mean())
print('Sales Representative Avg Salary', raw_data.loc[(raw_data['Department'] == 'Sales') & (raw_data['JobRole'] == 'Sales Representative'), 'MonthlyIncome'].mean())

# Research and Development
raw_data.loc[raw_data['Department'] == 'Research & Development', 'JobRole'].unique()
print('Research Scientist Avg Salary', raw_data.loc[(raw_data['Department'] == 'Research & Development') & (raw_data['JobRole'] == 'Research Scientist'), 'MonthlyIncome'].mean())
print('Laboratory Technician Avg Salary', raw_data.loc[(raw_data['Department'] == 'Research & Development') & (raw_data['JobRole'] == 'Laboratory Technician'), 'MonthlyIncome'].mean())
print('Manufacturing Director Avg Salary', raw_data.loc[(raw_data['Department'] == 'Research & Development') & (raw_data['JobRole'] == 'Manufacturing Director'), 'MonthlyIncome'].mean())
print('Healthcare Representative Avg Salary', raw_data.loc[(raw_data['Department'] == 'Research & Development') & (raw_data['JobRole'] == 'Healthcare Representative'), 'MonthlyIncome'].mean())
print('Research Director Avg Salary', raw_data.loc[(raw_data['Department'] == 'Research & Development') & (raw_data['JobRole'] == 'Research Director'), 'MonthlyIncome'].mean())
print('Manager Avg Salary', raw_data.loc[(raw_data['Department'] == 'Research & Development') & (raw_data['JobRole'] == 'Manager'), 'MonthlyIncome'].mean())

# Human Resources
raw_data.loc[raw_data['Department'] == 'Human Resources', 'JobRole'].unique()
print('Human Resources Avg Salary', raw_data.loc[(raw_data['Department'] == 'Human Resources') & (raw_data['JobRole'] == 'Human Resources'), 'MonthlyIncome'].mean())
print('Manager Avg Salary', raw_data.loc[(raw_data['Department'] == 'Human Resources') & (raw_data['JobRole'] == 'Manager'), 'MonthlyIncome'].mean())



# Job level
data['JobLevel'].unique()
print('level 1 avg monthly salary is', data.loc[data['JobLevel']==1,'MonthlyIncome'].mean())
print('level 2 avg monthly salary is', data.loc[data['JobLevel']==2,'MonthlyIncome'].mean())
print('level 3 avg monthly salary is', data.loc[data['JobLevel']==3,'MonthlyIncome'].mean())
print('level 4 avg monthly salary is', data.loc[data['JobLevel']==4,'MonthlyIncome'].mean())
print('level 5 avg monthly salary is', data.loc[data['JobLevel']==5,'MonthlyIncome'].mean())

# JobLevel dummy
JobLevel = data['JobLevel'].copy()
JobLevel[data['JobLevel']==1] = 'job_level_1'
JobLevel[data['JobLevel']==2] = 'job_level_2'
JobLevel[data['JobLevel']==3] = 'job_level_3'
JobLevel[data['JobLevel']==4] = 'job_level_4'
JobLevel[data['JobLevel']==5] = 'job_level_5'
data['JobLevel'] = JobLevel
data = one_hot(data, column='JobLevel')
print(data[['JobLevel','job_level_1','job_level_2','job_level_3','job_level_4','job_level_5']])
data.drop(['JobLevel'], axis=1, inplace=True)



# Education field
pprint(raw_data['EducationField'].unique().tolist())

data['job_match'] = 0
data['job_not_match'] = 0
data['job_unclear'] = 0

job_match = ((data['EducationField'] == 'Life Sciences') & (data['Research & Development'] == 1))
job_match = job_match | ((data['EducationField'] == 'Technical Degree') & (data['Research & Development'] == 1))
job_match = job_match | ((data['EducationField'] == 'Medical') & (data['Research & Development'] == 1))
job_match = job_match | ((data['EducationField'] == 'Marketing') & (data['Sales'] == 1))
job_match = job_match | ((data['EducationField'] == 'Human Resources') & (data['Human Resources'] == 1))

job_unclear = data['EducationField'] == 'Other'

job_not_match = (job_match == False) & (job_unclear == False)

data.loc[job_match, 'job_match'] = 1
data.loc[job_not_match, 'job_not_match'] = 1
data.loc[job_unclear, 'job_unclear'] = 1

temp = data[['Sales','Research & Development','Human Resources','EducationField','job_match','job_not_match','job_unclear']].copy()
print(temp)

temp['department'] = np.nan
temp.loc[temp['Sales']==1, 'department'] = 'S'
temp.loc[temp['Research & Development']==1, 'department'] = 'RD'
temp.loc[temp['Human Resources']==1, 'department'] = 'HR'
temp['edu_field+department'] = temp['EducationField'] + ' + ' + temp['department']

print('job match education field is', temp.loc[temp['job_match']==1,'edu_field+department'].unique())
print('job not match education field is', temp.loc[temp['job_not_match']==1,'edu_field+department'].unique())
print('job unclear education field is', temp.loc[temp['job_unclear']==1,'edu_field+department'].unique())

data.drop('EducationField', axis=1, inplace=True)

# Gender
raw_data['Gender'].unique()
data = one_hot(data, column='Gender',seperate_col=False, discete_dict={'Male':1, 'Female':0})
data = data.rename(columns={'Gender':'Male'})
pd.DataFrame(pd.concat([raw_data['Gender'], data['Male']], axis=1))
data['Male'] = data['Male'].astype(np.int64)

# Marital status
raw_data['MaritalStatus'].unique()
data = one_hot(data, column='MaritalStatus')
pd.DataFrame(pd.concat([raw_data['MaritalStatus'], data[['Single','Married','Divorced']]], axis=1))
data.drop('MaritalStatus', axis=1, inplace=True)

# Over 18
raw_data['Over18'].unique()
data.drop('Over18', axis=1, inplace=True)
assert 'Over18' not in data.columns.tolist()

# Over time
raw_data['OverTime'].unique()
data = one_hot(data, column='OverTime',seperate_col=False, discete_dict={'Yes':1, 'No':0})
pd.DataFrame(pd.concat([raw_data['OverTime'], data['OverTime']], axis=1))
data['OverTime'] = data['OverTime'].astype(np.int64)
data.info()
print(data.columns)
len(data.columns)



# step 2: descriptive analysis
data_stats_describe = data.describe(percentiles=[0.01,0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).transpose()
data_stats_describe['sum'] = data_stats_describe['mean'] * data_stats_describe['count']
print(data_stats_describe)

data_stats_describe.to_excel('data_stats_describe.xlsx')

# Correlation
corr_df = data.corr()
print(corr_df)
corr_df.to_excel('corr_analyst_df.xlsx')
# delete not very useful features
data.drop(['EmployeeCount','StandardHours'], axis=1, inplace=True)
data.drop(['EmployeeNumber'], axis=1, inplace=True)
data.drop('JobRole', axis=1, inplace=True)
data.info()

data.drop(['manager','middle','basic'], axis=1, inplace=True)

# again correlation
corr_df = data.corr()
print(corr_df)
# update correlation file
corr_df.to_excel('corr_analyst_df.xlsx')

# find pairs with the largest correlation
num_rows = corr_df.shape[0]
num_cols = corr_df.shape[1]
corr_dict = dict()
for i in range(num_rows):
    for j in range(num_cols):
        if i >= j:
            continue
        else:
            corr_dict[(corr_df.columns[j], corr_df.index[i])] = [corr_df.iloc[i, j]]

corr_pair_ranking_df = pd.DataFrame(corr_dict).transpose()
corr_pair_ranking_df = corr_pair_ranking_df.reset_index()
corr_pair_ranking_df.columns = ['f1', 'f2', 'corr_value']
corr_pair_ranking_df = corr_pair_ranking_df.sort_values('corr_value')
print(corr_pair_ranking_df)

corr_pair_ranking_df.to_excel('corr_pair_ranking_df.xlsx')


# data visualization
# distribution
fig = data.hist(bins=20, figsize = (20, 10))
plt.show()

data.columns.tolist()


# figures
# Department
raw_data['Department'].unique()

CurrentEmployed_Sales = data.loc[data['CurrentEmployee']==1, 'Sales'].sum()
CurrentEmployed_Research_Development = data.loc[data['CurrentEmployee']==1, 'Research & Development'].sum()
CurrentEmployed_Human_Resources = data.loc[data['CurrentEmployee']==1, 'Human Resources'].sum()


CurrentEmployed_Department = pd.Series([CurrentEmployed_Sales,
                                      CurrentEmployed_Research_Development,
                                      CurrentEmployed_Human_Resources],
                                     index=['Sales', 'Research & Development', 'Human Resources'])

NoCurrentEmployed_Sales = data.loc[data['CurrentEmployee']==0, 'Sales'].sum()
NoCurrentEmployed_Research_Development = data.loc[data['CurrentEmployee']==0, 'Research & Development'].sum()
NoCurrentEmployed_Human_Resources = data.loc[data['CurrentEmployee']==0, 'Human Resources'].sum()


NoCurrentEmployed_Department = pd.Series([NoCurrentEmployed_Sales,
                                      NoCurrentEmployed_Research_Development,
                                      NoCurrentEmployed_Human_Resources],
                                     index=['Sales', 'Research & Development', 'Human Resources'])


# plot
df_vis = pd.DataFrame([CurrentEmployed_Department, NoCurrentEmployed_Department])
df_vis.index = ['CurrentEmployee','NoCurrentEmployee']
df_vis.plot(kind='bar', stacked=True, figsize=(15,8))
plt.xticks(rotation=0)
plt.show()


# Over time
# generate Series
CurrentEmployed_Overtime = data[data['CurrentEmployee']==1]['OverTime'].value_counts()
NoCurrentEmployed_Overtime = data[data['CurrentEmployee']==0]['OverTime'].value_counts()

# plot
df_vis = pd.DataFrame([CurrentEmployed_Overtime,NoCurrentEmployed_Overtime])
df_vis.index = ['CurrentEmployee','NoCurrentEmployee']
df_vis.plot(kind='bar', stacked=True, figsize=(15,8))
plt.xticks(rotation=0)
plt.show()



# job_level
# generate Series
CurrentEmployed_JobLevel_1 = data.loc[data['CurrentEmployee']==1, 'job_level_1'].sum()
CurrentEmployed_JobLevel_2 = data.loc[data['CurrentEmployee']==1, 'job_level_2'].sum()
CurrentEmployed_JobLevel_3 = data.loc[data['CurrentEmployee']==1, 'job_level_3'].sum()
CurrentEmployed_JobLevel_4 = data.loc[data['CurrentEmployee']==1, 'job_level_4'].sum()
CurrentEmployed_JobLevel_5 = data.loc[data['CurrentEmployee']==1, 'job_level_5'].sum()
CurrentEmployed_JobLevel = pd.Series([CurrentEmployed_JobLevel_1,
                                      CurrentEmployed_JobLevel_2,
                                      CurrentEmployed_JobLevel_3,
                                      CurrentEmployed_JobLevel_4,
                                      CurrentEmployed_JobLevel_5],
                                     index=['job_level_1','job_level_2','job_level_3','job_level_4','job_level_5'])

NoCurrentEmployed_JobLevel_1 = data.loc[data['CurrentEmployee']==0, 'job_level_1'].sum()
NoCurrentEmployed_JobLevel_2 = data.loc[data['CurrentEmployee']==0, 'job_level_2'].sum()
NoCurrentEmployed_JobLevel_3 = data.loc[data['CurrentEmployee']==0, 'job_level_3'].sum()
NoCurrentEmployed_JobLevel_4 = data.loc[data['CurrentEmployee']==0, 'job_level_4'].sum()
NoCurrentEmployed_JobLevel_5 = data.loc[data['CurrentEmployee']==0, 'job_level_5'].sum()
NoCurrentEmployed_JobLevel = pd.Series([NoCurrentEmployed_JobLevel_1,
                                        NoCurrentEmployed_JobLevel_2,
                                        NoCurrentEmployed_JobLevel_3,
                                        NoCurrentEmployed_JobLevel_4,
                                        NoCurrentEmployed_JobLevel_5],
                                        index=['job_level_1','job_level_2','job_level_3','job_level_4','job_level_5'])

# plot
df_vis = pd.DataFrame([CurrentEmployed_JobLevel,NoCurrentEmployed_JobLevel])
df_vis.index = ['CurrentEmployee','NoCurrentEmployee']
df_vis.plot(kind='bar', stacked=True, figsize=(15,8))
plt.xticks(rotation=0)
plt.show()


# Over time
# generate Series
CurrentEmployed_Overtime = data[data['CurrentEmployee']==1]['OverTime'].value_counts()
NoCurrentEmployed_Overtime = data[data['CurrentEmployee']==0]['OverTime'].value_counts()

# plot
df_vis = pd.DataFrame([CurrentEmployed_Overtime,NoCurrentEmployed_Overtime])
df_vis.index = ['CurrentEmployee','NoCurrentEmployee']
df_vis.plot(kind='bar', stacked=True, figsize=(15,8))
plt.xticks(rotation=0)
plt.show()


# Gender
# generate Series
CurrentEmployed_Male = data[data['CurrentEmployee']==1]['Male'].value_counts()
NoCurrentEmployed_Male = data[data['CurrentEmployee']==0]['Male'].value_counts()

# plot
df_vis = pd.DataFrame([CurrentEmployed_Male,NoCurrentEmployed_Male])
df_vis.index = ['CurrentEmployee','NoCurrentEmployee']
df_vis.plot(kind='bar', stacked=True, figsize=(15,8))
plt.xticks(rotation=0)




# Job match or not
# generate Series
CurrentEmployed_JobMatch = data.loc[data['CurrentEmployee'] == 1, 'job_match'].sum()
CurrentEmployed_JobNotMatch = data.loc[data['CurrentEmployee'] == 1, 'job_not_match'].sum()
CurrentEmployed_JobUnclear = data.loc[data['CurrentEmployee'] == 1, 'job_unclear'].sum()

CurrentEmployed_JobMatch = pd.Series([CurrentEmployed_JobMatch,
                                      CurrentEmployed_JobNotMatch,
                                      CurrentEmployed_JobUnclear],
                                     index=['job_match', 'job_not_match', 'job_unclear'])

NoCurrentEmployed_JobMatch = data.loc[data['CurrentEmployee'] == 0, 'job_match'].sum()
NoCurrentEmployed_JobNotMatch = data.loc[data['CurrentEmployee'] == 0, 'job_not_match'].sum()
NoCurrentEmployed_JobUnclear = data.loc[data['CurrentEmployee'] == 0, 'job_unclear'].sum()

NoCurrentEmployed_JobMatch = pd.Series([NoCurrentEmployed_JobMatch,
                                        NoCurrentEmployed_JobNotMatch,
                                        NoCurrentEmployed_JobUnclear],
                                       index=['job_match', 'job_not_match', 'job_unclear'])

# plot
df_vis = pd.DataFrame([CurrentEmployed_JobMatch, NoCurrentEmployed_JobMatch])
df_vis.index = ['CurrentEmployee', 'NoCurrentEmployee']
df_vis.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.xticks(rotation=0)
plt.legend(('job_match', 'job_not_match', 'job_unclear'), loc='center left', fontsize=15, bbox_to_anchor=(1.01, 0.5))
plt.show()



PromotionYears = []
Levels = sorted(data['JobSatisfaction'].unique())

for Level in Levels:
    PromotionYears.append(data.loc[data['JobSatisfaction']==Level,'YearsSinceLastPromotion'])

plt.boxplot(x = PromotionYears,
            patch_artist=True,
            labels = ['1','2','3','4'],
            boxprops = {'color':'black','facecolor':'#9999ff'},
            flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},
            medianprops = {'linestyle':'--','color':'orange'})

plt.show()





# Education dummy
EducationLevel = data['Education'].copy()
EducationLevel[data['Education']==1] = 'Below College'
EducationLevel[data['Education']==2] = 'College'
EducationLevel[data['Education']==3] = 'Bachelor'
EducationLevel[data['Education']==4] = 'Master'
EducationLevel[data['Education']==5] = 'Doctor'

data['Education'] = EducationLevel
data = one_hot(data, column='Education')
print(data[['Education', 'Below College','College','Bachelor','Master','Doctor']])

# Education
# generate Series
CurrentEmployed_Below_College = data.loc[data['CurrentEmployee'] == 1, 'Below College'].sum()
CurrentEmployed_College = data.loc[data['CurrentEmployee'] == 1, 'College'].sum()
CurrentEmployed_Bachelor = data.loc[data['CurrentEmployee'] == 1, 'Bachelor'].sum()
CurrentEmployed_Master = data.loc[data['CurrentEmployee'] == 1, 'Master'].sum()
CurrentEmployed_Doctor = data.loc[data['CurrentEmployee'] == 1, 'Doctor'].sum()

CurrentEmployed_Education = pd.Series([CurrentEmployed_Below_College,
                                       CurrentEmployed_College,
                                       CurrentEmployed_Bachelor,
                                       CurrentEmployed_Master,
                                       CurrentEmployed_Doctor],
                                      index=['Below College', 'College', 'Bachelor', 'Master', 'Doctor'])

NoCurrentEmployed_Below_College = data.loc[data['CurrentEmployee'] == 0, 'Below College'].sum()
NoCurrentEmployed_College = data.loc[data['CurrentEmployee'] == 0, 'College'].sum()
NoCurrentEmployed_Bachelor = data.loc[data['CurrentEmployee'] == 0, 'Bachelor'].sum()
NoCurrentEmployed_Master = data.loc[data['CurrentEmployee'] == 0, 'Master'].sum()
NoCurrentEmployed_Doctor = data.loc[data['CurrentEmployee'] == 0, 'Doctor'].sum()

NoCurrentEmployed_Education = pd.Series([NoCurrentEmployed_Below_College,
                                         NoCurrentEmployed_College,
                                         NoCurrentEmployed_Bachelor,
                                         NoCurrentEmployed_Master,
                                         NoCurrentEmployed_Doctor],
                                        index=['Below College', 'College', 'Bachelor', 'Master', 'Doctor'])

# plot
df_vis = pd.DataFrame([CurrentEmployed_Education, NoCurrentEmployed_Education])
df_vis.index = ['CurrentEmployee', 'NoCurrentEmployee']
df_vis.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.xticks(rotation=0)
plt.legend(('Below College', 'College', 'Bachelor', 'Master', 'Doctor'), loc='center left', fontsize=15,
           bbox_to_anchor=(1.01, 0.5))




# worklikfebalance dummy, bad, good, better, best
WorkLifeBalanceLevel = data['WorkLifeBalance'].copy()
WorkLifeBalanceLevel[data['WorkLifeBalance']==1] = 'bad'
WorkLifeBalanceLevel[data['WorkLifeBalance']==2] = 'good'
WorkLifeBalanceLevel[data['WorkLifeBalance']==3] = 'better'
WorkLifeBalanceLevel[data['WorkLifeBalance']==4] = 'best'

data['WorkLifeBalance'] = WorkLifeBalanceLevel
data = one_hot(data, column='WorkLifeBalance')
print(data[['WorkLifeBalance','bad','good','better','best']])


# generate Series
CurrentEmployed_Bad = data.loc[data['CurrentEmployee'] == 1, 'bad'].sum()
CurrentEmployed_Good = data.loc[data['CurrentEmployee'] == 1, 'good'].sum()
CurrentEmployed_Better = data.loc[data['CurrentEmployee'] == 1, 'better'].sum()
CurrentEmployed_Best = data.loc[data['CurrentEmployee'] == 1, 'best'].sum()


CurrentEmployed_WorkLifeBalance = pd.Series([CurrentEmployed_Bad,
                                             CurrentEmployed_Good,
                                             CurrentEmployed_Better,
                                             CurrentEmployed_Best],
                                      index=['bad', 'good', 'better','best'])

NoCurrentEmployed_Bad = data.loc[data['CurrentEmployee'] == 0, 'bad'].sum()
NoCurrentEmployed_Good = data.loc[data['CurrentEmployee'] == 0, 'good'].sum()
NoCurrentEmployed_Better = data.loc[data['CurrentEmployee'] == 0, 'better'].sum()
NoCurrentEmployed_Best = data.loc[data['CurrentEmployee'] == 0, 'best'].sum()

NoCurrentEmployed_WorkLifeBalance = pd.Series([NoCurrentEmployed_Bad,
                                             NoCurrentEmployed_Good,
                                             NoCurrentEmployed_Better,
                                             NoCurrentEmployed_Best],
                                      index=['bad', 'good', 'better','best'])

# plot
df_vis = pd.DataFrame([CurrentEmployed_WorkLifeBalance,NoCurrentEmployed_WorkLifeBalance])
df_vis.index = ['CurrentEmployee', 'NoCurrentEmployee']
df_vis.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.xticks(rotation=0)
plt.legend(('bad', 'good', 'better','best'), loc='center left', fontsize=15,
           bbox_to_anchor=(1.01, 0.5))
plt.show()



# worklifebalance compare with gender
# generate Series
Male_Bad = data.loc[data['Male'] == 1, 'bad'].sum()
Male_Good = data.loc[data['Male'] == 1, 'good'].sum()
Male_Better = data.loc[data['Male'] == 1, 'better'].sum()
Male_Best = data.loc[data['Male'] == 1, 'best'].sum()


Male_WorkLifeBalance = pd.Series([Male_Bad,Male_Good,Male_Better,Male_Best],index=['bad', 'good', 'better','best'])

Female_Bad = data.loc[data['Male'] == 0, 'bad'].sum()
Female_Good = data.loc[data['Male'] == 0, 'good'].sum()
Female_Better = data.loc[data['Male'] == 0, 'better'].sum()
Female_Best = data.loc[data['Male'] == 0, 'best'].sum()

Female_WorkLifeBalance = pd.Series([Female_Bad,Female_Good,Female_Better,Female_Best],index=['bad', 'good', 'better','best'])

# plot
df_vis = pd.DataFrame([Male_WorkLifeBalance,Female_WorkLifeBalance ])
df_vis.index = ['Male', 'Female']
df_vis.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.xticks(rotation=0)
plt.legend(('bad', 'good', 'better','best'), loc='center left', fontsize=15,
           bbox_to_anchor=(1.01, 0.5))
plt.title('WorkLifeBalance & Gender')
plt.show()




# MonthlyIncome + Age
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(data[data['CurrentEmployee']==1]['MonthlyIncome'],data[data['CurrentEmployee']==1]['Age'],c='green',s=40)
ax.scatter(data[data['CurrentEmployee']==0]['MonthlyIncome'],data[data['CurrentEmployee']==0]['Age'],c='red',s=40)
plt.xlabel('MonthlyIncome',fontsize=15)
plt.ylabel('Age',fontsize=15)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
ax.legend(('CurrentEmployeed','NoCurrentEmployeed'),scatterpoints=1,loc='center left',fontsize=15, bbox_to_anchor=(1.01, 0.5))
plt.show()



# YearsSinceLastPromotion + Age
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(data[data['CurrentEmployee']==1]['YearsSinceLastPromotion'],data[data['CurrentEmployee']==1]['Age'],c='green',s=40)
ax.scatter(data[data['CurrentEmployee']==0]['YearsSinceLastPromotion'],data[data['CurrentEmployee']==0]['Age'],c='red',s=40)
plt.xlabel('YearsSinceLastPromotion',fontsize=15)
plt.ylabel('Age',fontsize=15)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
ax.legend(('CurrentEmployeed','NoCurrentEmployeed'),scatterpoints=1,loc='center left',fontsize=15, bbox_to_anchor=(1.01, 0.5))
plt.show()



# MonthlyIncome + YearsAtCompany
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(data[data['CurrentEmployee']==1]['MonthlyIncome'],data[data['CurrentEmployee']==1]['YearsAtCompany'],c='green',s=40)
ax.scatter(data[data['CurrentEmployee']==0]['MonthlyIncome'],data[data['CurrentEmployee']==0]['YearsAtCompany'],c='red',s=40)
plt.xlabel('MonthlyIncome',fontsize=15)
plt.ylabel('YearsAtCompany',fontsize=15)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
ax.legend(('CurrentEmployeed','NoCurrentEmployeed'),scatterpoints=1,loc='center left',fontsize=15, bbox_to_anchor=(1.01, 0.5))
plt.show()



# MontylyIncome + StockOptionLevel
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(data[data['CurrentEmployee']==1]['MonthlyIncome'],data[data['CurrentEmployee']==1]['StockOptionLevel'],c='green',s=40)
ax.scatter(data[data['CurrentEmployee']==0]['MonthlyIncome'],data[data['CurrentEmployee']==0]['StockOptionLevel'],c='red',s=40)
plt.xlabel('MonthlyIncome',fontsize=15)
plt.ylabel('StockOptionLevel',fontsize=15)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
ax.legend(('CurrentEmployeed','NoCurrentEmployeed'),scatterpoints=1,loc='center left',fontsize=15, bbox_to_anchor=(1.01, 0.5))
plt.show()



# MonthlyIncome + TrainingTimesLastYear
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(data[data['CurrentEmployee']==1]['MonthlyIncome'],data[data['CurrentEmployee']==1]['TrainingTimesLastYear'],c='green',s=40)
ax.scatter(data[data['CurrentEmployee']==0]['MonthlyIncome'],data[data['CurrentEmployee']==0]['TrainingTimesLastYear'],c='red',s=40)
plt.xlabel('MonthlyIncome',fontsize=15)
plt.ylabel('TrainingTimesLastYear',fontsize=15)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
ax.legend(('CurrentEmployeed','NoCurrentEmployeed'),scatterpoints=1,loc='center left',fontsize=15, bbox_to_anchor=(1.01, 0.5))
plt.show()


# MonthlyIncome + PercentSalaryHike
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(data[data['CurrentEmployee']==1]['MonthlyIncome'],data[data['CurrentEmployee']==1]['PercentSalaryHike'],c='green',s=40)
ax.scatter(data[data['CurrentEmployee']==0]['MonthlyIncome'],data[data['CurrentEmployee']==0]['PercentSalaryHike'],c='red',s=40)
plt.xlabel('MonthlyIncome',fontsize=15)
plt.ylabel('PercentSalaryHike',fontsize=15)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width, box.height])
ax.legend(('CurrentEmployeed','NoCurrentEmployeed'),scatterpoints=1,loc='center left',fontsize=15, bbox_to_anchor=(1.01, 0.5))
plt.show()



#box plot PercentSalaryHike
fig = plt.figure()
ax = plt.subplot()
CurrentEmployed_PercentSalaryHike = data.loc[data['CurrentEmployee'] == 1, 'PercentSalaryHike']
NoCurrentEmployed_PercentSalaryHike = data.loc[data['CurrentEmployee'] == 0, 'PercentSalaryHike']

ax.boxplot([CurrentEmployed_PercentSalaryHike,NoCurrentEmployed_PercentSalaryHike], notch=True)
ax.set_xticks([1,2])
ax.set_xticklabels(['CurrentEmployee', 'NoCurrentEmployee'])
plt.title('PercentSalaryHike')
plt.grid(axis='y')
plt.show()




##### Clustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
plt.rcParams['figure.figsize'] = (16, 9)
import pandas as pd
import numpy as np
import random
plt.style.use('ggplot')

# raw_data = pd.read_csv('Hackathon_Data_2018_Rev.csv')
# data = raw_data.copy()

f1 = data['MonthlyIncome'].values
# f2 = data['Male'].values
f2 = data['JobSatisfaction'].values
X = np.array(list(zip(f1, f2)))

y = []
for i in range(1470):
    y.append(random.randint(1,3))
y = np.array(y)

kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
C = kmeans.cluster_centers_

fig = plt.figure()
plt.scatter(f1, f2, c=y, s=7)
# plt.scatter(C[:, 0], C[:, 1], marker='v', c='black',s=1000)
# ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
plt.show()



# build models, target: Current Employee
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection  import StratifiedKFold
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder

print(data.shape)



# Cross validation
# Cut training data and test data
train_test_ratio = 0.8

sss = StratifiedShuffleSplit(n_splits=2, test_size=(1 - train_test_ratio), random_state=123)
for train_index, test_index in sss.split(data.drop('CurrentEmployee', axis=1), data['CurrentEmployee']):
    train_data = data.iloc[train_index, :]
    test_data = data.iloc[test_index, :]

# divide target and features
train_x = train_data.drop('CurrentEmployee', axis=1)
train_y = train_data['CurrentEmployee']

test_x = test_data.drop('CurrentEmployee', axis=1)
test_y = test_data['CurrentEmployee']


# this is a biased dataset
print('train set - current employee sample num is %d, no current employee sample is %d.' % (train_y[train_y==1].shape[0], train_y[train_y==0].shape[0]))
print('test set - current employee sample num is %d, no current employee sample is %d.' % (test_y[test_y==1].shape[0], test_y[test_y==0].shape[0]))

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt',class_weight='balanced')
clf = clf.fit(train_x.values, train_y.values)

features = pd.DataFrame()
features['feature'] = train_x.columns
features['importance'] = clf.feature_importances_
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(15,8))
plt.xlabel('Importance')
plt.yticks(fontsize=7)
plt.show()


model = SelectFromModel(clf, prefit=True)
train_x_reduced = model.transform(train_x)
test_x_reduced = model.transform(test_x)

print(train_x_reduced.shape)
print(test_x_reduced.shape)


# improve parameters
parameter_grid = {
     'max_depth' : [3, 4, 5, 6, 7, 8],
     'n_estimators': [50, 100, 200, 300],
     'max_features': ['sqrt', 'log2']
}

forest = RandomForestClassifier(class_weight='balanced')

skf = StratifiedKFold(n_splits=10)

grid_search = GridSearchCV(forest,
                           scoring='accuracy',
                           param_grid=parameter_grid,
                           cv=skf)

grid_search.fit(train_x.values, train_y.values)

# model = grid_search
parameters = grid_search.best_params_

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))

y_train_pred = grid_search.predict(train_x.values)
y_test_pred = grid_search.predict(test_x.values)
confusion_matrix(test_y.values, y_test_pred)

print('test:\n',classification_report(test_y.values, y_test_pred))
print('train:\n',classification_report(train_y.values, y_train_pred))



# re-train model, change scoring parameter from accuracy to roc_auc
parameter_grid = {
     'max_depth' : [3, 4, 5, 7, 8, 9, 10],
     'n_estimators': [50, 100, 200],
     'max_features': [0.1, 0.2, 0.3, 0.4, 0.5]
}


forest = RandomForestClassifier(class_weight='balanced')

skf = StratifiedKFold(n_splits=3)

grid_search_biased = GridSearchCV(forest,
                               scoring='roc_auc',
                               param_grid=parameter_grid,
                               cv=skf)

grid_search_biased.fit(train_x.values, train_y.values)

# model = grid_search
parameters = grid_search_biased.best_params_

print('Best score: {}'.format(grid_search_biased.best_score_))
print('Best parameters: {}'.format(grid_search_biased.best_params_))

y_train_pred_roc = grid_search_biased.predict(train_x.values)
y_test_pred_roc = grid_search_biased.predict(test_x.values)
confusion_matrix(test_y.values, y_test_pred_roc)

print('test:\n',classification_report(test_y.values, y_test_pred_roc))
print('train:\n',classification_report(train_y.values, y_train_pred_roc))

print('class 0 num is', test_y[test_y==0].shape)
print('class 1 num is', test_y[test_y==1].shape)


df = pd.DataFrame(test_y)
df[0] = 0
df[1] = 0
df.loc[df['CurrentEmployee']==0, 0] = 1
df.loc[df['CurrentEmployee']==1, 1] = 1
test_y_roc = df[[0,1]].values

test_y_score = grid_search.predict_proba(test_x.values)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(test_y_roc[:, i], test_y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(test_y_roc.ravel(), test_y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# ROC
plt.figure()
lw = 2
plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()





## Decision Tree

from sklearn import tree
import pandas as pd
import graphviz
import numpy as np

train_test_ratio = 0.8

model_data = data[['CurrentEmployee','MonthlyIncome', 'TotalWorkingYears', 'Age', 'OverTime', 'DailyRate',
                   'DistanceFromHome', 'MonthlyRate', 'HourlyRate', 'YearsAtCompany',
                   'StockOptionLevel', 'JobSatisfaction','PercentSalaryHike',
                   'YearsWithCurrManager', 'NumCompaniesWorked', 'YearsInCurrentRole',
                   'EnvironmentSatisfaction', 'YearsSinceLastPromotion', 'TrainingTimesLastYear',
                   'RelationshipSatisfaction', 'WorkLifeBalance', 'Education', 'JobInvolvement']]


sss = StratifiedShuffleSplit(n_splits=2, test_size=(1 - train_test_ratio), random_state=123)
for train_index, test_index in sss.split(model_data.drop('CurrentEmployee', axis=1), model_data['CurrentEmployee']):
    train_data = model_data.iloc[train_index, :]
    test_data = model_data.iloc[test_index, :]

# divide target and features
train_x = train_data.drop('CurrentEmployee', axis=1)
train_y = train_data['CurrentEmployee']
train_x.columns.tolist()
# type(train_target)

test_x = test_data.drop('CurrentEmployee', axis=1)
test_y = test_data['CurrentEmployee']


clf = tree.DecisionTreeClassifier(max_leaf_nodes=8, class_weight='balanced')
clf = clf.fit(train_x.values, train_y.values)

## try different values of the parameters
clf = tree.DecisionTreeClassifier(max_leaf_nodes=8)
clf = clf.fit(train_x.values, train_y.values)

clf = tree.DecisionTreeClassifier(max_leaf_nodes=12, class_weight='balanced')
clf = clf.fit(train_x.values, train_y.values)


model_data.columns.tolist()


train_features = train_x.columns.tolist()
train_target = np.array((0,1))
train_target = train_target.astype('<U10')

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=train_features,
                         class_names=train_target,
                         filled=True, rounded=True,
                         special_characters=True)

graph = graphviz.Source(dot_data)
graph.render('employee3')




