#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pickle


# # READ DATA

# In[2]:


df_adverspend = pd.read_csv("advertising_spend_data.csv",encoding='utf-8')


# In[3]:


df_adverspend


# In[4]:


df_adverspend.columns = ['date','facebook','email','search','brand sem intent google','affiliate','email_blast','pinterest','referral']


# In[5]:


df_adverspend


# In[8]:


df_adverspend = df_adverspend.drop(index = 0)


# In[9]:


df_adverspend


# In[10]:


df_rep = pd.read_pickle("customer_service_reps")


# In[ ]:


df_rep.columns


# In[12]:


df_engagement = pd.read_pickle("engagement")


# In[13]:


df_engagement


# In[14]:


df_subscribers = pd.read_pickle("subscribers")


# In[15]:


df_subscribers


# In[16]:


df_subscribers.loc[2,:]


# # EDA

# In[17]:


df_rep.groupby('customer_service_rep_id').nunique()


# In[19]:


df_rep.groupby('subid').nunique()


# In[18]:


df_subscribers.groupby('subid').nunique()


# In[22]:


df_subscribers.groupby('age').nunique()


# In[33]:


def clean_age(data):
    age = data['age']
    
    if age >= 70:
        return "Others"
    elif age >= 50:
        return "Elderly(50-70)"
    elif age >= 35:
        return "Mid-aged(35-50)"
    elif age >= 18:
        return "Youth(18-35)"
    elif age >= 0:
        return "Teenagers(<18)"
    else:
        return "Others"

    return age


# In[34]:


df_subscribers['age_group'] = df_subscribers.apply(clean_age,axis=1)


# In[35]:


df_subscribers.groupby('age_group').count()


# In[20]:


def trans_data(data):
    results = {}
    keys = set()
    for key in data:
        first = key[0]
        second = key[1]
        keys.add(second)
        if first not in results:
            results[first] = {}
        results[first][second] = data[key]
    for key in results:
        for inner_key in keys:
            if inner_key not in results[key]:
                results[key][inner_key] = 0
    return results


# In[43]:


merged = pd.merge(df_subscribers,df_rep,on='subid', how='inner')


# In[46]:


merged.columns


# In[237]:


get_num = lambda x:x.iloc[1]


# In[93]:


abtest


# # AB TESTING

# converted ( customer who completed trial, payment period is not 0, not canceling or requesting refund )

# In[132]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels


# ### 7 days v.s. 14 days

# In[413]:


a_num = df_rep[df_rep['num_trial_days']== 14]


# In[414]:


b_num = df_rep[df_rep['num_trial_days']== 7]


# In[415]:


a_num


# In[416]:


b_num


# In[417]:


A_purchased = a_num[a_num['payment_period']!= 0]


# In[418]:


A_purchased = A_purchased[A_purchased['trial_completed_TF']== True]


# In[139]:


A_purchased = A_purchased[A_purchased['current_sub_TF']== True]


# In[419]:


A_purchased


# In[141]:


a_num.groupby('subid').nunique()


# In[424]:


A_purchased.groupby('subid').nunique()


# In[427]:


B_purchased = b_num[b_num['payment_period']!= 0]


# In[428]:


B_purchased = B_purchased[B_purchased['trial_completed_TF']== True]


# In[145]:


B_purchased = B_purchased[B_purchased['current_sub_TF']== True]


# In[429]:


b_num.groupby('subid').nunique()


# In[430]:


B_purchased.groupby('subid').nunique()


# In[425]:


p_A = 523596/1281127


# In[426]:


p_A


# In[431]:


p_B = 35314/64043


# In[432]:


p_B


# In[433]:


diff = p_B - p_A


# In[434]:


diff


# In[437]:


z = diff /np.sqrt(((p_B * (1-p_B)/64043)+(p_A * (1-p_A)/1281127)))


# In[438]:


z


# H0: 14days = 7days (14 days is better than 7)

# H1: 7days > 14days (7 days is better than 14)

# In[449]:


def opt_sample_size(basline, diff, power, sig_level):

    standard_norm = scs.norm(0, 1)
    Z_beta = standard_norm.ppf(power)
    Z_alpha = standard_norm.ppf(1-sig_level/2)
    overall_prob = (basline*2+diff) / 2

    opt_N = (2 * overall_prob * (1 - overall_prob) * (Z_beta + Z_alpha)**2
             / diff**2)

    return opt_N


# In[450]:


opt_sample_size(p_A, p_B - p_A, 0.8, 0.05)


# In[451]:


opt_sample_size(p_A1, p_B1 - p_A1, 0.8, 0.05)


# ### high v.s. low

# In[336]:


df_subscribers.groupby('plan_type').count()


# In[367]:


a_num1 = df_subscribers[df_subscribers['plan_type']== 'high_uae_14_day_trial']


# In[456]:


a_num1.groupby('subid').nunique()


# In[398]:


b_num1 = df_subscribers[df_subscribers['plan_type']== 'base_uae_14_day_trial']


# In[489]:


a_num1.groupby('subid').nunique()


# In[490]:


A_purchased1.groupby('cancel_before_trial_end').nunique()


# In[497]:


A_purchased1 = a_num1[a_num1['paid_TF'] == True ]


# In[498]:


A_purchased1 = A_purchased1[A_purchased1['refund_after_trial_TF'] == False ]


# In[499]:


B_purchased1 = b_num1[b_num1['paid_TF'] == True ]


# In[500]:


B_purchased1 = B_purchased1[B_purchased1['refund_after_trial_TF'] == False ]


# In[501]:


A_purchased1.groupby('subid').nunique()


# In[502]:


B_purchased1.groupby('subid').nunique()


# In[503]:


p_A1 = 97/325


# In[504]:


p_A1


# In[505]:


p_B1 = 82558/227096


# In[506]:


p_B1


# In[507]:


diff1 = p_B1 - p_A1


# In[508]:


diff1


# In[509]:


z = diff1 /np.sqrt(((p_B1 * (1-p_B1)/227096)+(p_A1 * (1-p_A1)/325)))


# In[510]:


z


# H0: high = low 

# H1: low > high 

# # CLUSTERING

# In[199]:


dummy_cluster = pd.get_dummies(data=df_subscribers, columns=['age_group','package_type','preferred_genre','intended_use','attribution_technical','plan_type'])


# In[200]:


dummy_cluster


# In[201]:


dummy_cluster = dummy_cluster.drop(['retarget_TF','male_TF','country','attribution_survey','op_sys','account_creation_date','creation_until_cancel_days','cancel_before_trial_end','trial_end_date','initial_credit_card_declined','revenue_net','join_fee','language','paid_TF', 'refund_after_trial_TF',
       'payment_type'],axis=1)


# In[202]:


dummy_cluster = dummy_cluster.drop(['attribution_technical_affiliate', 'attribution_technical_appstore',
       'attribution_technical_bing', 'attribution_technical_bing_organic',
       'attribution_technical_brand sem intent bing',
       'attribution_technical_brand sem intent google',
       'attribution_technical_content_greatist',
       'attribution_technical_criteo', 'attribution_technical_direct_mail',
       'attribution_technical_discovery', 'attribution_technical_display',
       'attribution_technical_email', 'attribution_technical_email_blast',
       'attribution_technical_facebook',
       'attribution_technical_facebook_organic',
       'attribution_technical_google_organic',
       'attribution_technical_influencer', 'attribution_technical_internal',
       'attribution_technical_organic', 'attribution_technical_other',
       'attribution_technical_ott', 'attribution_technical_pinterest',
       'attribution_technical_pinterest_organic',
       'attribution_technical_playstore', 'attribution_technical_podcast',
       'attribution_technical_quora', 'attribution_technical_referral',
       'attribution_technical_samsung', 'attribution_technical_search',
       'attribution_technical_tv', 'attribution_technical_twitter',
       'attribution_technical_vod', 'attribution_technical_youtube',],axis=1)


# In[203]:


dummy_cluster.columns


# In[204]:


dummy_cluster = dummy_cluster.drop(['subid'],axis=1)


# In[205]:


print(dummy_cluster.corr())


# In[206]:


dummy_cluster = dummy_cluster.drop(['num_ideal_streaming_services'],axis=1)


# In[207]:


dummy_cluster['num_weekly_services_utilized'] = dummy_cluster['num_weekly_services_utilized'].fillna(3)


# In[208]:


dummy_cluster['weekly_consumption_hour'] = dummy_cluster['weekly_consumption_hour'].fillna(23.625)


# In[210]:


dummy_cluster = dummy_cluster.dropna(axis=0, how='any')


# In[211]:


dummy_cluster = dummy_cluster.drop(['age'],axis=1)


# In[212]:


dummy_cluster


# !! forgot to clean age

# In[197]:


import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

distortions=[]
for i in range(2,12):
    km = skc.KMeans(n_clusters=i, random_state=9)
    y_pred=km.fit_predict(dummy_cluster)
    print(i)
    print(metrics.calinski_harabaz_score(dummy_cluster, y_pred))
    distortions.append(km.inertia_)

plt.plot(range(2,12),distortions,marker='o')
plt.xlabel('组数量')
plt.ylabel("SSE")
plt.show()


# right version

# In[213]:


import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

distortions=[]
for i in range(2,12):
    km = skc.KMeans(n_clusters=i, random_state=9)
    y_pred=km.fit_predict(dummy_cluster)
    print(i)
    print(metrics.calinski_harabaz_score(dummy_cluster, y_pred))
    distortions.append(km.inertia_)

plt.plot(range(2,12),distortions,marker='o')
plt.xlabel('组数量')
plt.ylabel("SSE")
plt.show()


# In[512]:


from scipy.cluster.vq import kmeans, vq
km = skc.KMeans(n_clusters=3, random_state=1)
y_pred=km.fit_predict(dummy_cluster)

r1=pd.Series(km.labels_).value_counts()
r2=pd.DataFrame(km.cluster_centers_)
r=pd.concat([r2,r1],axis=1)
r.columns=list(dummy_cluster.columns)+[u'count']

r=r.T
r.round(3)


# In[513]:


r.to_csv('r_final2.csv')


# # CHURN

# In[218]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[227]:


formodel = pd.merge(df_subscribers,df_rep,on='subid', how='inner')


# In[228]:


formodel


# In[231]:


formodel.columns


# In[233]:


topredict.columns


# In[238]:


get_num = lambda x:x.iloc[0]


# In[239]:


formodel = formodel.groupby('subid').agg(
    repid=('customer_service_rep_id', get_num),
    billing_channel=('billing_channel',get_num),
    num_trial_days=('num_trial_days',get_num),
    revenue_net_1month=('revenue_net_1month',get_num),
    payment_period=('payment_period','max'),
    package_type=('package_type',get_num),
    weekly_consumption_hour=('weekly_consumption_hour',get_num),
    male_TF=('male_TF',get_num),
    country=('country',get_num),
    op_sys=('op_sys',get_num),
    renew = ('renew',get_num),
    months_per_bill_period=('months_per_bill_period',get_num),
    creation_until_cancel_days=('creation_until_cancel_days',get_num),
    cancel_before_trial_end=('cancel_before_trial_end',get_num),
    revenue_net=('revenue_net','sum'),
    paid_TF=('paid_TF',get_num),
    refund_after_trial_TF=('refund_after_trial_TF',get_num),
    payment_type=('payment_type',get_num),
    current_sub_TF=('current_sub_TF',get_num),
    monthly_price=('monthly_price',get_num),
    discount_price=('discount_price',get_num)).reset_index()


# In[240]:


X = formodel.drop(['repid', 'subid', 'current_sub_TF'],axis=1)
y = formodel['current_sub_TF']
y = y.replace('True',1)
y = y.replace('False',0)


# In[274]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[241]:


X = formodel.drop(['repid','current_sub_TF','renew','payment_type','subid','country','num_trial_days'],axis=1)


# In[242]:


X.columns


# In[243]:


X=pd.get_dummies(X, columns=['billing_channel','op_sys','package_type'],drop_first=True)


# In[244]:


X


# In[246]:


def convert_tf(x):
    if x == 'True':
        return 1
    else:
        return 0


# In[247]:


X['gender']= X['male_TF'].apply(convert_tf)


# In[248]:


X['cancel']= X['cancel_before_trial_end'].apply(convert_tf)


# In[249]:


X['paid']= X['paid_TF'].apply(convert_tf)


# In[250]:


X['refund']= X['refund_after_trial_TF'].apply(convert_tf)


# In[251]:


X = X.drop(['male_TF','cancel_before_trial_end','paid_TF','refund_after_trial_TF'],axis=1)


# In[252]:


topredict = pd.merge(df_subscribers,df_rep,on='subid', how='inner')


# In[533]:


topredict


# In[253]:


topredict1 = topredict[topredict['retarget_TF'] == False]


# In[534]:


topredict1 = topredict


# In[535]:


topredict1


# In[666]:


topredict2 = topredict1.groupby('subid').agg(
    
    repid=('customer_service_rep_id', get_num),
    billing_channel=('billing_channel',get_num),
    num_trial_days=('num_trial_days',get_num),
    revenue_net_1month=('revenue_net_1month',get_num),
    payment_period=('payment_period','max'),
    package_type=('package_type',get_num),
    weekly_consumption_hour=('weekly_consumption_hour',get_num),
    male_TF=('male_TF',get_num),
    country=('country',get_num),
    op_sys=('op_sys',get_num),
    renew = ('renew',get_num),
    months_per_bill_period=('months_per_bill_period',get_num),
    creation_until_cancel_days=('creation_until_cancel_days',get_num),
    cancel_before_trial_end=('cancel_before_trial_end',get_num),
    revenue_net=('revenue_net','sum'),
    paid_TF=('paid_TF',get_num),
    refund_after_trial_TF=('refund_after_trial_TF',get_num),
    payment_type=('payment_type',get_num),
    current_sub_TF=('current_sub_TF',get_num),
    monthly_price=('monthly_price',get_num),
    discount_price=('discount_price',get_num),
    attribution_technical=('attribution_technical',get_num)).reset_index()


# In[667]:


topredict2 = topredict2.drop(['subid','country','num_trial_days'],axis=1)


# In[668]:


topredict2['gender']= topredict2['male_TF'].apply(convert_tf)


# In[669]:


topredict2['cancel']= topredict2['cancel_before_trial_end'].apply(convert_tf)


# In[670]:


topredict2['paid']= topredict2['paid_TF'].apply(convert_tf)


# In[671]:


topredict2['refund']= topredict2['refund_after_trial_TF'].apply(convert_tf)


# In[672]:


topredict2 = topredict2.drop(['male_TF','cancel_before_trial_end','paid_TF','refund_after_trial_TF'],axis=1)


# In[673]:


topredict2=pd.get_dummies(topredict2, columns=['billing_channel','op_sys','package_type'],drop_first=True)


# In[674]:


topredict2


# In[675]:


topredict2.fillna(0)


# In[546]:


predictX = topredict2.drop(['repid','current_sub_TF','renew','payment_type'],axis=1)


# In[547]:


X = X.fillna(0)


# In[548]:


X


# In[552]:


predictX


# ## logistic regression

# In[267]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.utils import check_random_state


# In[549]:


from sklearn.model_selection import GridSearchCV
scale = RobustScaler().fit(X_train)

param_test = {
        'C':np.logspace(-5, -1, 50)
    }

estimator = LogisticRegression(solver='lbfgs',max_iter = 10000)
gsearch = GridSearchCV(estimator, param_grid = param_test, cv=10)
gsearch.fit(X_train,y_train)
print(gsearch.best_params_)


# In[550]:


from sklearn.metrics import accuracy_score
clf_logistic = LogisticRegression(C = gsearch.best_params_['C'],solver='lbfgs',max_iter = 10000)
clf_logistic.fit(X_train,y_train)
y_pred = clf_logistic.predict(X_test)
accuracy_score(y_test,y_pred)


# In[551]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[553]:


predictX.isnull().sum()


# In[676]:


topredict2['predictions_logistic'] = clf_logistic.predict(X)


# In[677]:


topredict2['churn_prob_logistic'] = clf_logistic.predict_proba(X)[:,1]


# In[678]:


topredict2.groupby('predictions_logistic').count()


# In[679]:


topredict2


# In[577]:


churn_prob_logistic = clf_logistic.predict_proba(X)[:,1]


# In[578]:


pd.DataFrame(churn_prob).describe()


# ## Decision Tree

# In[565]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

param_test = {
        'max_depth': [2,3,4,5,6]
    }

estimator = DecisionTreeClassifier(random_state = 0)
gsearch = GridSearchCV(estimator, param_grid = param_test, cv=10)
gsearch.fit(X_train,y_train)
print(gsearch.best_params_)


# In[566]:


clf = DecisionTreeClassifier(max_depth = gsearch.best_params_['max_depth'],random_state = 0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test,y_pred)


# In[567]:


print(classification_report(y_test, y_pred))


# In[568]:


topredict2['predictions_DT'] = clf.predict(X)


# In[576]:


topredict2['churn_prob_dt'] = clf.predict_proba(X)[:,1]


# In[569]:


topredict2.groupby('predictions_DT').count()


# In[570]:


clf.feature_importances_.round(3)


# In[571]:


import graphviz


# In[572]:


from sklearn.tree import export_graphviz
dot_data = export_graphviz(clf, out_file=None,feature_names = X_train.columns) 
graph = graphviz.Source(dot_data) 
graph.render("TREE_Final")


# In[574]:


churn_prob_dt = clf.predict_proba(X)[:,1]


# In[575]:


pd.DataFrame(churn_prob_dt).describe()


# ## RandomForest

# In[580]:


from sklearn.ensemble import RandomForestClassifier

# grid search
param_test = {
    'n_estimators':[10,50,100,200],
    'max_depth':[2,3,4,5,6]
    }
estimator = RandomForestClassifier()
gsearch = GridSearchCV(estimator, param_grid = param_test, cv=10)
gsearch.fit(X_train,y_train)
print('best score is:',str(gsearch.best_score_))
print('best params are:',str(gsearch.best_params_))


# In[581]:


clfrand = RandomForestClassifier(max_depth = gsearch.best_params_['max_depth'], n_estimators = gsearch.best_params_['n_estimators'])
clfrand.fit(X_train,y_train)
y_pred = clfrand.predict(X_test)
accuracy_score(y_test,y_pred)


# In[582]:


print(classification_report(y_test, y_pred))


# In[583]:


clfrand.feature_importances_.round(3)


# In[584]:


topredict2['predictions_RF'] = clfrand.predict(X)


# In[585]:


topredict2.groupby('predictions_RF').count()


# In[587]:


churn_prob_dt= clfrand.predict_proba(X)[:,1]


# In[589]:


topredict2['churn_prob_dt'] = clfrand.predict_proba(X)[:,1]


# In[590]:


pd.DataFrame(churn_prob_dt).describe()


# ## GBDT

# In[591]:


from sklearn.ensemble import GradientBoostingClassifier

# grid search
param_test = {
    'n_estimators':[10, 50, 100, 200],
    'max_depth':[2,3,4,5,6]
    }
estimator = GradientBoostingClassifier()
gsearch = GridSearchCV(estimator, param_grid = param_test, cv=10)
gsearch.fit(X_train,y_train)
print('best score is:',round(float(gsearch.best_score_), 2))
print('best params are:',str(gsearch.best_params_))


# In[592]:


# Train GBDT model and do prediction
clfgbdt = GradientBoostingClassifier(max_depth = gsearch.best_params_['max_depth'], n_estimators = gsearch.best_params_['n_estimators'])
clfgbdt.fit(X_train,y_train)
y_pred = clfgbdt.predict(X_test)
round(accuracy_score(y_test,y_pred),4)


# In[593]:


print(classification_report(y_test, y_pred))


# In[594]:


clfgbdt.feature_importances_.round(3)


# In[680]:


topredict2['predictions_GBDT'] = clfgbdt.predict(X)


# In[681]:


topredict2.groupby('predictions_GBDT').count()


# In[682]:


churn_prob_gbdt= clfgbdt.predict_proba(X)[:,1]


# In[683]:


pd.DataFrame(churn_prob_gbdt).describe()


# In[684]:


topredict2['churn_prob_gbdt'] = clfgbdt.predict_proba(X)[:,1]


# # CLV

# In[604]:


merged = pd.merge(df_subscribers,df_rep,on='subid', how='inner')


# In[609]:


merged['month'],merged['year'] = merged['account_creation_date_x'].dt.month, merged['account_creation_date_x'].dt.year


# In[611]:


merged['converted'] = np.where((merged['cancel_before_trial_end'] == False) | (merged['revenue_net'] == 0) | (merged['refund_after_trial_TF'] == True),
                                    False, True)


# In[612]:


merged


# In[616]:


merged.groupby(['converted']).subid.count()


# In[637]:


acq = merged.loc[:,['subid','month','cancel_before_trial_end','converted', 'attribution_technical']]


# In[638]:


acq


# In[639]:


acq = acq.drop('cancel_before_trial_end', axis = 1)


# In[640]:


# change acquisition channel to match the spend data
acq['attribution_technical'] = np.where(acq['attribution_technical'].isin(df_adverspend.columns), sub_acq['attribution_technical'], 
                                                 np.where(acq['attribution_technical'].str.contains("organic"), "organic", 'other'))

acq.head()


# In[641]:


monthacq = acq[acq['converted'] == True].groupby('month').subid.agg('count')

plt.figure(figsize = (10,4))
plt.plot(month_sub_acq.index.astype('str'), month_sub_acq)
plt.show()


# In[647]:


monthacq


# In[645]:


spend_total = acq.set_index('month').sum(axis = 1)


# In[652]:


spend_total = spend_total.groupby('month').sum()


# In[655]:


spend_total.to_csv('cac.csv')


# In[687]:


cac = pd.read_csv("cac.csv",encoding='utf-8')


# In[688]:


acq.to_csv('acq.csv')


# In[690]:


cac


# In[660]:


r = 0.1 / 3


# In[755]:


r


# In[756]:


topredict2['future_rev'] = (topredict2['monthly_price'])*((1+r)/(1+r-(1-topredict2['churn_prob_gbdt']))) - (topredict2['monthly_price'])


# In[686]:


topredict2.columns


# In[744]:


forclv


# In[745]:


forclv['clv'] = forclv['revenue_net'] + forclv['future_rev'] - forclv['cac']


# In[760]:


forclv


# In[775]:


forclv.to_csv('forclv.csv')


# In[777]:


clv = forclv['clv']


# In[773]:


clv_mean


# In[774]:


clv_median


# In[778]:


pd.DataFrame(churn_prob_gbdt).describe()


# In[785]:


type(churn_prob_gbdt)


# In[786]:


clv = np.array(clv)


# In[ ]:




