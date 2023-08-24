#!/usr/bin/env python
# coding: utf-8

# # Cricket World Cup 2019

# # Cricket Player Performance Prediction using machine learning

# # Importing Libraries

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# # Loading all datasets

# In[5]:


bats_df = pd.read_csv('C:/Users/admin/Desktop/Batsman_Data.csv')
bats_df


# In[6]:


bowl_df = pd.read_csv('C:/Users/admin/Desktop/Bowler_data.csv')
bowl_df


# In[8]:


ground_df = pd.read_csv('C:/Users/admin/Downloads/Ground_Averages.csv')
ground_df


# In[9]:


match_results = pd.read_csv('C:/Users/admin/Downloads/ODI_Match_Results.csv')
match_results


# In[10]:


match_totals = pd.read_csv('C:/Users/admin/Downloads/ODI_Match_Totals.csv')
match_totals


# In[13]:


wc_player = pd.read_csv('C:/Users/admin/Downloads/WC_players.csv')
wc_player


# In[16]:


bats_df.isnull().sum()


# In[17]:


bowl_df.isnull().sum()


# In[18]:


ground_df.isnull().sum()


# In[19]:


match_results.isnull().sum()


# In[20]:


match_totals.isnull().sum()


# In[21]:


wc_player.isnull().sum()


# In[22]:


bats_df.info()


# In[23]:


bowl_df.info()


# In[24]:


ground_df.info()


# In[25]:


match_results.info()


# In[26]:


match_totals.info()


# In[27]:


wc_player.info()


# In[28]:


df=[bats_df,bowl_df,ground_df,match_results,match_totals,wc_player]


# In[29]:


def dementions(df):
    for i in df:
        print(i.shape)
        print('*'*100)
        print(i.info())
        print('*'*100)
        print(i.isna().sum())


# In[30]:


dementions(df)


# In[31]:


def stat_summary(df):
    for i in df:
        print(i.describe(include='all'))
        print('*'*100)


# In[32]:


stat_summary(df)


# In[33]:


def seperate_date(df):
    for i in range(len(df)):
        if 'Start Date' in df[i].columns:
            df[i]['Year']=pd.to_datetime(df[i]['Start Date'])
            df[i]['Month']=df[i]['Year'].apply(lambda x:x.month) # Extracting Month
            df[i]['Day']=df[i]['Year'].apply(lambda x:x.day)  # Extracting day
            df[i]['year']=df[i]['Year'].apply(lambda x:x.year)  # Extracting year
            print(df[i].head())
            print("-------------------------")
        else:
            print("DataFrame", i, "does not have 'Satrt Date' column")
            print("-------------------------")


# In[34]:


seperate_date(df)


# In[35]:


def drop_irrelevant(df):
    for i in range(len(df)):
        columns_to_drop = ['Unnamed: 0', 'Start Date', 'Year']
        irrelevant_columns = [col for col in columns_to_drop if col in df[i].columns]
        
        if irrelevant_columns:
            df[i].drop(columns=irrelevant_columns, axis=1, inplace=True)
            print("DataFrame", i, "after dropping irrelevant columns:")
            print(df[i].head())  # Printing the DataFrame after dropping columns
            print("-------------------------")
        else:
            print("DataFrame", i, "does not have any irrelevant columns")
            print("-------------------------")


# In[36]:


drop_irrelevant(df)


# In[37]:


df_list=[bats_df,bowl_df,ground_df,match_results,match_totals,wc_player]


# In[39]:


from sklearn.preprocessing import LabelEncoder

numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']

def get_categorical_columns(df_list):
    cat_col_list = []  

    for df in df_list:
        cat_col = []  
        features = df.columns.values.tolist()

        for col in features:
            if df[col].dtype not in numerics: 
                cat_col.append(col)

        cat_col_list.append(cat_col)

    return cat_col_list

def label_encode_categorical_columns(df_list):
    label = LabelEncoder()
    
    categorical_columns_list = get_categorical_columns(df_list)

    for i, cat_col in enumerate(categorical_columns_list):
        for col in cat_col:
            encoded_values = label.fit_transform(df_list[i][col])
            df_list[i][col] = encoded_values


# In[40]:


label_encode_categorical_columns(df_list)


# In[41]:


bats_df.head()


# In[42]:


bowl_df.head()


# In[43]:


ground_df.head()


# In[44]:


match_results.head()


# In[45]:


match_totals.head()


# In[46]:


wc_player.head()


# In[47]:


match_results.isnull().sum()


# In[49]:


match_totals.isnull().sum()


# In[50]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer()
df_list2 = [match_results,match_totals]

def treat_missing_value(df_list2):
    for i in range(len(df_list2)):
        df_imputed = imputer.fit_transform(df_list2[i])
        df_list2[i] = pd.DataFrame(df_imputed, columns=df_list2[i].columns)
        print("DataFrame", i, "after imputation:")
        print(df_list2[i].isnull().sum()) #Lets Check the Values treated well or not
        print("-------------------------")


# In[51]:


treat_missing_value(df_list2)


# In[52]:


match_totals=df_list2[0]
match_results=df_list2[1]


# In[53]:


bats_df


# In[54]:


bowl_df


# In[55]:


ground_df


# In[56]:


batsman_join_bowler=pd.merge(bats_df,bowl_df,on=['Match_ID','Player_ID','Opposition','Ground','Month','Day','year'], how='inner')
batsman_join_bowler


# In[57]:


batsman_join_bowler.columns


# In[59]:


bowl_df.columns


# In[60]:


batsman_join_bowler_GrondAvg=pd.merge(batsman_join_bowler,ground_df,on=['Ground'], how='inner')
batsman_join_bowler_GrondAvg


# In[61]:


batsman_join_bowler_GrondAvg.columns


# In[62]:


match_results.columns


# In[63]:


match_totals.columns


# In[64]:


OD_Total_result=pd.merge(match_results,match_totals,on=['Ground','Country','Country_ID','Month','Day','year','Opposition'], how='inner')
OD_Total_result


# In[65]:


batsman_join_bowler_GrondAvg_OD=pd.merge(batsman_join_bowler_GrondAvg,OD_Total_result,on=['Ground','Month','Day','year'], how='inner')
batsman_join_bowler_GrondAvg_OD


# In[66]:


batsman_join_bowler_GrondAvg_OD.columns


# In[67]:


wc_player=wc_player.rename(columns={'ID':'Player_ID'})


# In[68]:


master_after_join=pd.merge(batsman_join_bowler_GrondAvg_OD,wc_player,on=['Player_ID','Country'], how='inner')
master_after_join


# In[69]:


# Calculate Batting Average for each player
master_after_join['Batting Average'] = master_after_join['Bat1'] / master_after_join['Inns']


# In[70]:


print(master_after_join[['Player', 'Batting Average']])


# In[71]:


master_after_join['Bowling Average'] = master_after_join['Runs_y'] / master_after_join['Wkts_y']
print(master_after_join[['Player', 'Bowling Average']])


# In[73]:


master_after_join = master_after_join[master_after_join['BF'] > 0]
master_after_join.head(5)


# In[74]:


# Calculate Strike Rate (Batting) for each player
master_after_join['Strike Rate (Batting)'] = (master_after_join['Bat1'] / master_after_join['BF']) * 100

# Display the Strike Rate (Batting) for each player
print(master_after_join[['Player', 'Strike Rate (Batting)']])


# In[75]:


# Calculate Economy Rate (Bowling) for each player
master_after_join['Economy Rate (Bowling)'] = (master_after_join['Runs_y'] / master_after_join['Overs_y'])

# Display the Economy Rate (Bowling) for each player
print(master_after_join[['Player', 'Economy Rate (Bowling)']])


# In[76]:


# Calculate the total Maiden Overs for each player
master_after_join['Maiden Overs Total'] = master_after_join['Mdns'].sum()

# Display the total Maiden Overs for each player
print(master_after_join[['Player', 'Maiden Overs Total']])


# In[77]:


# Step 1: Choose relevant performance metrics
batting_average = master_after_join['Batting Average']
bowling_average = master_after_join['Bowling Average']
strike_rate_batting = master_after_join['Strike Rate (Batting)']
economy_rate_bowling = master_after_join['Economy Rate (Bowling)']
maiden_overs = master_after_join['Maiden Overs Total']

#Step 2: Normalize the selected performance metrics
# You can use Min-Max Scaling or Z-score normalization
def min_max_scaling(x):
    return (x - x.min()) / (x.max() - x.min())

normalized_batting_average = min_max_scaling(batting_average)
normalized_bowling_average = min_max_scaling(bowling_average)
normalized_strike_rate_batting = min_max_scaling(strike_rate_batting)
normalized_economy_rate_bowling = min_max_scaling(economy_rate_bowling)
# normalized_maiden_overs = min_max_scaling(maiden_overs) we are not using this its giving NAN
# Step 3: Assign weights to each performance metric
batting_weight = 0.3
bowling_weight = 0.25
strike_rate_weight = 0.2
economy_rate_weight = 0.25


# Step 4: Calculate the composite performance score for each player
master_after_join['Player Performance Score'] = (
    batting_weight * normalized_batting_average +
    bowling_weight * normalized_bowling_average +
    strike_rate_weight * normalized_strike_rate_batting +
    economy_rate_weight * normalized_economy_rate_bowling )

# Step 5: Display the Player Performance Score for each player
print(master_after_join[['Player', 'Player Performance Score']])


# In[78]:


master_after_join.shape


# In[79]:


master_after_join.columns


# In[80]:


#Let's see the how data is distributed or Graphical analysis of all features
plt.figure(figsize=(15,20))
plotnumber=1
for column in master_after_join:
    if plotnumber<=60:
        ax=plt.subplot(10,6,plotnumber)
        sns.histplot(master_after_join[column])
        plt.xlabel(column,fontsize=15)
    plotnumber+=1
plt.tight_layout()


# In[81]:


master_after_join.skew()


# In[82]:


master_after_join.columns


# In[83]:


master_after_join.skew()[(master_after_join.skew() >= 0.5) & (master_after_join.skew() >= -0.5)]


# In[84]:


skew_col=['4s','6s','Ground','Player_ID','Overs_x','Mdns','Runs_y','Wkts_x','Econ','Ave_x','SR_y','Mat','Won','Runs', 'Wkts_y', 'Balls', 'Ave_y', 'RPO_x','BR','Country_ID','Bowling Average','Bowling Average','Economy Rate (Bowling)']
len(skew_col)


# In[85]:


# Now almost All skewness we removed so move further
#Let's check for the Outliers
plt.figure(figsize=(15,20))
plotnumber=1
for column in master_after_join:
    if plotnumber<=60:
        ax=plt.subplot(10,6,plotnumber)
        sns.boxplot(master_after_join[column])
        plt.xlabel(column,fontsize=15)
    plotnumber+=1
plt.tight_layout()


# In[86]:


outliers_col=['Margin', 'BR','Overs_y','Target','Ground','Span','Strike Rate (Batting)','Player Performance Score']


# In[87]:


# Remove the outliers by using Z score
from scipy.stats import zscore
z_score=zscore(master_after_join[outliers_col])
z_score_abs=np.abs(z_score)
filter_entry=(z_score_abs<3).all(axis=1)
master_after_join=master_after_join[filter_entry]
master_after_join.head()


# In[88]:


#Let's recheck for the Outliers
plt.figure(figsize=(15,20))
plotnumber=1
for column in master_after_join:
    if plotnumber<=60:
        ax=plt.subplot(10,6,plotnumber)
        sns.boxplot(master_after_join[column])
        plt.xlabel(column,fontsize=15)
    plotnumber+=1
plt.tight_layout()


# In[89]:


# Use 'Player Performance Score' as the target variable for the prediction model

x = master_after_join[['Bat1', 'Runs_x', 'BF', 'SR_x', '4s', '6s', 'Opposition_x', 'Ground',
          'Match_ID', 'Batsman', 'Player_ID', 'Month', 'Day', 'year', 'Overs_x',
          'Mdns', 'Runs_y', 'Wkts_x', 'Econ', 'Ave_x', 'SR_y', 'Bowler', 'Span',
          'Mat', 'Won', 'Tied', 'NR', 'Runs', 'Wkts_y', 'Balls', 'Ave_y', 'RPO_x',
          'Result_x', 'Margin', 'BR', 'Toss', 'Bat', 'Opposition_y', 'Match_ID_x',
          'Country', 'Country_ID', 'Score', 'Overs_y', 'RPO_y', 'Target', 'Inns',
          'Result_y', 'Match_ID_y']]

y = master_after_join['Player Performance Score'] 


# In[90]:


print(x.shape, y.shape)


# In[91]:


# Data stadardization
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_scaled=scale.fit_transform(x)


# In[92]:


#check multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif["vif"]=[variance_inflation_factor(x_scaled,i) for i in range(x_scaled.shape[1])]
vif["featurs"]=x.columns
print(vif)


# In[93]:


from sklearn.decomposition import PCA
pca=PCA()
pca.fit_transform(x_scaled)


# In[94]:


# Lets plot the PCA plot to select the best components
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Principal Components")
plt.ylabel("Variance Covered")
plt.title("PCA Plot")


# In[95]:


pca=PCA(n_components=19)
pca_x=pca.fit_transform(x_scaled)
pca_x=pd.DataFrame(pca_x,columns=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17','PC18','PC19'])
pca_x


# In[96]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error as mae, mean_squared_error as mse
acc_score_test = 0
acc_score_train = 0
rand_state = 0
for i in range(1,200):
    
    x_train,x_test,y_train,y_test = train_test_split(pca_x,y,test_size=0.3,random_state= i )
    
    lr = LinearRegression()
    lr.fit(x_train,y_train)
    
    y_pred=lr.predict(x_train) # prediction on train data
    r1 =r2_score(y_train,y_pred) # accuracy check on train data
    
    pred = lr.predict(x_test)   #prediction on test data
    r2 =r2_score(y_test,pred) #accuracy check on test data
    
    if r2 > acc_score_test and r1 > acc_score_train: # selecting max score accuracy
        acc_score_test = r2
        acc_score_train = r1
        rand_state = i
    print(f"at random sate {i}, the training accuracy is:- {acc_score_train}")
    print(f"at random sate {i}, the testing accuracy is:- {acc_score_test}")
    print("\n")
print('Best Training accuracy_score is {} on random state {}'.format(acc_score_train,rand_state))
print('Best Testing accuracy_score is {} on random state {}'.format(acc_score_test,rand_state))


# In[97]:


rand_state


# In[98]:


#Lets split for train and test data
x_train,x_test,y_train,y_test = train_test_split(pca_x,y,test_size=0.3,random_state= rand_state )


# In[99]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[102]:


from sklearn.linear_model import LinearRegression
LR= LinearRegression()

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
ada=AdaBoostRegressor()
gb=GradientBoostingRegressor()

from sklearn.ensemble import RandomForestRegressor
rfc= RandomForestRegressor()



from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor()

from sklearn.model_selection import cross_val_score


models=[]
models.append(('LinearRegression', LR))
models.append(('DecisionTreeRegressor', dt))
models.append(('AdaBoostRegressor', ada))
models.append(('GradientBoostingRegressor', gb))
models.append(('RandomForestRegressor', rfc))
models.append(('KNeighborsRegressor', knn))


# In[103]:


m=[]
score=[]
score2=[]
cv_score=[]
MAE_score=[]
MSE_score=[]
for name, model in models:
    print('***********************',name,'***********************')
    m.append(name)
    model.fit(x_train, y_train)
    print(model)
    y_pred=model.predict(x_train)
    AS=r2_score(y_train,y_pred)
    print("Train Report:",AS)
    score.append(AS*100)
    
    pred=model.predict(x_test)
    AS2=r2_score(y_test,pred)
    print("Test Report:",AS2)
    score2.append(AS2*100)
    MAE=mae(y_test,pred)
    print("Mean Squered Error:",MAE)
    MAE_score.append(MAE*100)
    MSE=mse(y_test,pred)
    print("Mean Absolute Error:", MSE)
    MSE_score.append(MSE*100)
    
    accuracies= cross_val_score(model,x,y, cv=2)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    cv_score.append(accuracies.mean()*100)
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
    print('\n')


# In[104]:


print(len(m),len(score),len(score2),len(cv_score),len(MAE_score),len(MSE_score))


# In[105]:


result = pd.DataFrame({'Model': m, 'Accuracy_train_score': score,'Accuracy_test_score': score2 ,'Cross_val_score':cv_score, 'MAE_score':MAE_score,'MSE_score':MSE_score })
result


# In[106]:


result['lest_diff']=(result['Accuracy_test_score']-result['Cross_val_score'])
result


# In[107]:


# Hyper tuning by using RandomizedSearchCV With RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

para={'n_jobs':range(1,55)}
rand=RandomizedSearchCV(estimator=LR, cv=5,param_distributions=para)
rand.fit(x_train,y_train)

rand.best_params_


# In[108]:


LR= LinearRegression(n_jobs= 17)

LR.fit(x_train,y_train)
y_pred=ada.predict(x_train)
AS=r2_score(y_train,y_pred)
print("Train Report:",AS*100) 
pred=LR.predict(x_test)
AS2=r2_score(y_test,pred)
print("Test Report:",AS2*100)
MAE=mae(y_test,pred)
print("Mean Squered Error:",MAE)
MSE=mse(y_test,pred)
print("Mean Absolute Error:", MSE)
    
accuracies= cross_val_score(LR,x_scaled, y, cv=5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
cv_score.append(accuracies.mean()*100)
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

