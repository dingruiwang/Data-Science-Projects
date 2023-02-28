#!/usr/bin/env python
# coding: utf-8

# In[161]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[162]:


train = pd.read_csv("training_SyncPatient.csv").dropna()
test = pd.read_csv("test_SyncPatient.csv").dropna()
data = pd.concat([train,test])


# ## EDA

# In[163]:


# cleaning: DOB to Age 
data["age"] = (2022 - data["YearOfBirth"]).astype(int)
data = data.drop("YearOfBirth",axis=1)
data = data.reset_index().drop("index",axis=1)
data


# In[164]:


# use training data for EDA
Patient = data.iloc[train.index,:]


# ### age 

# In[165]:


diabete = Patient[Patient["DMIndicator"]==1]
non_diabete = Patient[Patient["DMIndicator"]==0]

Patient.loc[Patient["DMIndicator"]==1,"DMIndicator"] = "diabete"
Patient.loc[Patient["DMIndicator"]==0,"DMIndicator"] = "non-diabete"


# In[191]:


sns.set(rc = {'figure.figsize':(6,4)})
bins = np.arange(0,110,5)
sns.histplot(diabete["age"], bins=bins,stat='density', label='diabete', ec='w');
sns.histplot(non_diabete["age"], bins=bins,stat='density', label='non_diabete', color='gold', alpha = 0.4, ec='w');
plt.title("diabete vs non-diabete in Age")
plt.legend();


# ### gender

# In[167]:


sns.set(rc = {'figure.figsize':(10,3)})
table = pd.crosstab(Patient.DMIndicator,Patient.Gender);
table.div(table.sum(1).astype(float), axis=0).plot(kind='barh',stacked=True);
plt.title('diabete vs non-diabete: Gender');
plt.legend(loc='best',bbox_to_anchor=(1, 1))
plt.xlabel('Percent');
plt.ylabel('DMIndicator');


# ### smoke condition

# In[168]:


SmokingStatus = pd.read_csv("training_SyncPatientSmokingStatus.csv")
SmokingStatus_desc = pd.read_csv("SyncSmokingStatus.csv")


# In[169]:


# merge 
Patient = Patient.merge(SmokingStatus,how="left",left_on="PatientGuid",right_on="PatientGuid")
Patient = Patient.merge(SmokingStatus_desc,how="left",left_on="SmokingStatusGuid",right_on="SmokingStatusGuid")

# check duplicates
print("duplicates before:",sum(Patient["PatientGuid"].duplicated()))

# keeping the most recent effective record 
Patient = Patient.sort_values(by="EffectiveYear",ascending=False)
Patient = Patient.drop_duplicates("PatientGuid")
Patient = Patient.sort_index().reset_index().drop("index",axis=1)

print("duplicates after:",sum(Patient["PatientGuid"].duplicated()))


# In[170]:


smoke = Patient[~Patient["SmokingStatusGuid"].isnull()]


# In[171]:


sns.set(rc = {'figure.figsize':(10,5)})
p = smoke.groupby('DMIndicator')['Description'].value_counts(normalize=True).mul(100).rename('percent').reset_index() .pipe((sns.catplot,'data'), x='DMIndicator',y='percent',hue='Description',kind='bar');

# overall, about 85% of patients in the train dataset are non-smoker, no matter if they have diabete
 
# the category definition is not very clear, such as no big diff between 0 cig per day(prev smoker) and not a current tabacco user, 
# so let's just break into two category: present smoker and non-smoker 


# In[172]:


non_smoker_category = ["5ABBAB35-836F-4F3E-8632-CE063828DA15","C12C2DB7-D31A-4514-88C0-42CBD339F764","1F3BFBBF-AB76-481B-B1E0-08A3689A54BC"]
smoker_category = ["FCD437AA-0451-4D8A-9396-B6F19D8B25E8","02116D5A-F26C-4A48-9A11-75AC21BC4FD3","2548BD83-03AE-4287-A578-FA170F39E32F","FA2B7AE4-4D14-4768-A8C7-55B5F0CDF4AF","DD01E545-D7AF-4F00-B248-9FD40010D81D"]

smoke.loc[:,"is_smoker"] = ""
smoke.loc[smoke["SmokingStatusGuid"].isin(non_smoker_category),"is_smoker"] = "smoker"
smoke.loc[smoke["SmokingStatusGuid"].isin(smoker_category),"is_smoker"] = "non-smoker"
smoke.loc[(~smoke["SmokingStatusGuid"].isin(smoker_category))&(~smoke["SmokingStatusGuid"].isin(non_smoker_category)),"is_smoker"] = np.nan
smoke['is_smoker'] = smoke['is_smoker'].fillna(method="ffill")


# In[198]:


sns.set(rc = {'figure.figsize':(10,3)})
table = pd.crosstab(smoke.DMIndicator,smoke.is_smoker);
table.div(table.sum(1).astype(float), axis=0).plot(kind='barh',stacked=True);
plt.title('diabete vs non-diabete: Smoke condition');
plt.legend(loc='best',bbox_to_anchor=(1, 1))
plt.xlabel('Percent');
plt.ylabel('DMIndicator');
# by breaking to binary category:smoker or non-smoker, there's little difference between diabete and non-diabete patients 


# ### transcript data

# In[174]:


transcript = pd.read_csv("training_SyncTranscript.csv")
# drop null or 0 values 
transcript = transcript[transcript["BMI"] != 0].drop(["HeartRate","PhysicianSpecialty"],axis=1)

# drop duplicates: keeping the most recent effective record 
transcript = transcript.sort_values(by="VisitYear",ascending=False)
transcript = transcript.drop_duplicates("PatientGuid")
transcript = transcript.sort_index().reset_index().drop("index",axis=1)

# impute null with mean 
transcript = transcript.fillna(transcript.mean())


# In[175]:


data = pd.merge(data,transcript,how="left",on="PatientGuid")


# In[196]:


diabete = data[data["DMIndicator"]==1]
non_diabete = data[data["DMIndicator"]==0]

sns.set(rc = {'figure.figsize':(6,4)})
bins = np.arange(0,70,5)
sns.histplot(diabete["BMI"], bins=bins,stat='density', label='diabete', ec='w');
sns.histplot(non_diabete["BMI"], bins=bins,stat='density', label='non_diabete', color='gold', alpha  = 0.4,ec='w');
plt.title("diabete vs non-diabete in BMI")
plt.legend();

## we can see that BMI of diabete patients is higher 


# In[195]:


non_diabete


# ### diagnosis data (icd9)

# In[83]:


def categorize_icd9code(code):
    icd9code = {    
        '272': 'Disorders of lipoid metabolism',
        '401': 'Essential hypertension',
        '585': 'Chronic renal failure',
        '715': 'Osteoarthrosis and allied disorders',
        '414': 'Other forms of chronic ischemic heart disease',
        '782': 'Symptoms involving skin and other integumentary tissue',
        '443': 'Other peripheral vascular disease',
        '428': 'Heart failure',
        '285': 'Other and unspecified anemias',
        '781': 'Symptoms involving nervous and musculoskeletal systems',
        '276': 'Disorders of fluid, electrolyte, and acid-base balance',
        '791': 'Nonspecific findings on examination of urine',
        'v03+v04': 'prophylactic vaccination and inoculation',
        '600': 'Hyperplasia of prostate',
        '715': 'certain conditions originating in the perinatal period',
        '716': 'Other and unspecified arthropathies',
        '496': 'Chronic airway obstruction, not elsewhere classified',
        '438': 'Late effects of cerebrovascular disease',
        '461': 'Acute sinusitis',
        '706': 'Diseases of sebaceous glands',
        '314': 'Hyperkinetic syndrome of childhood',
        '300':'Neurotic disorders'
    }
    code = code.split('.')[0]
    if ('V03' in code.upper()) or ('V04' in code.upper()): return 'prophylactic vaccination and inoculation'
    elif ('E' in code.upper()) or ('V' in code.upper()): return 'Other Supplementary'
    elif int(code) == 272: return 'Disorders of lipoid metabolism'
    elif int(code) == 401: return 'Essential hypertension'
    elif int(code) == 585: return 'Chronic renal failure'
    elif int(code) == 715: return 'Osteoarthrosis and allied disorders'
    elif int(code) == 414: return 'Other forms of chronic ischemic heart disease'
    elif int(code) == 782: return 'Symptoms involving skin and other integumentary tissue'
    elif int(code) == 443: return 'Other peripheral vascular disease'
    elif int(code) == 428: return 'Heart failure'
    elif int(code) == 285: return 'Other and unspecified anemias'
    elif int(code) == 781: return 'Symptoms involving nervous and musculoskeletal systems'
    elif int(code) == 276: return 'Disorders of fluid, electrolyte, and acid-base balance'
    elif int(code) == 791: return 'Nonspecific findings on examination of urine'
    elif int(code) == 600: return 'Hyperplasia of prostate'
    elif int(code) == 715: return 'certain conditions originating in the perinatal period'
    elif int(code) == 716: return 'Other and unspecified arthropathies'
    elif int(code) == 496: return 'Chronic airway obstruction, not elsewhere classified'
    elif int(code) == 438: return 'Late effects of cerebrovascular disease'   
    elif int(code) == 461: return 'Acute sinusitis'
    elif int(code) == 706: return 'Diseases of sebaceous glands'
    elif int(code) == 314: return 'Hyperkinetic syndrome of childhood'
    elif int(code) == 300: return 'Neurotic disorders' 
    else: return 'Other Comorbidity categories'


# In[84]:


Diagnosis = pd.read_csv("training_SyncDiagnosis.csv")
Diagnosis['ICD9CodeCategory'] = Diagnosis.ICD9Code.apply(lambda x:categorize_icd9code(x))


# In[85]:


# aggregate and get dummies of ICD9CodeCategory 
diagnosis_agg = Diagnosis[['ICD9CodeCategory']]
diagnosis_agg.index = Diagnosis.PatientGuid
diagnosis_agg = pd.get_dummies(diagnosis_agg,prefix='',prefix_sep='').reset_index().groupby('PatientGuid').sum()
data = data.set_index("PatientGuid").join(diagnosis_agg).reset_index()


# In[86]:


data.columns


# In[ ]:





# In[ ]:





# In[49]:


'''
# I think we need to take state population into consideration
# but not sure if should do some adjustment for onehot encoding of state 

pop = pd.read_csv("pop_2012.csv")
pat_by_state = Patient.groupby("State").agg(len).reset_index()
pat_pop_merged = pat_by_state.merge(pop,how="inner",left_on="State",right_on="Code")[["State_x","PatientGuid","POP_2012"]]
pat_pop_merged.loc[:,"patient density"] = pat_pop_merged["PatientGuid"]/pat_pop_merged["POP_2012"]
pat_pop_merged = pat_pop_merged.sort_values("patient density",ascending=False)

for i in range(len(onehot)):
    for j in range(len(pat_pop_merged)):
        if onehot.loc[i,"State_"+str(pat_pop_merged.loc[j,"State_x"])] == 1:
            onehot.loc[i,"State_"+str(pat_pop_merged.loc[j,"State_x"])] = pat_pop_merged.loc[j,"patient density"]
            
'''


# ## Feature Engineering

# In[87]:


# covert DMIndicator back to boolean values 
train.loc[train["DMIndicator"]=="diabete","DMIndicator"] = 1
train.loc[train["DMIndicator"]=="non-diabete","DMIndicator"] = 0

train.loc[train["Gender"]=="M","Gender"] = 1
train.loc[train["Gender"]=="F","Gender"] = 0

train["DMIndicator"] = train["DMIndicator"].astype(float)


# ### One hot encoding

# In[88]:


onehot = pd.get_dummies(data[["State","Gender"]],columns=["State","Gender"])

            
data = pd.concat([data,onehot],axis=1)
data = data.drop(["State","Gender"],axis=1)


# In[96]:


X_cols = ['age',  'Gender_F', 'Gender_M', 'Height', 'Weight', 'BMI', 'SystolicBP', 'DiastolicBP',
       'RespiratoryRate', 'Temperature',
       'Acute sinusitis',
       'Chronic airway obstruction, not elsewhere classified',
       'Chronic renal failure', 'Diseases of sebaceous glands',
       'Disorders of fluid, electrolyte, and acid-base balance',
       'Disorders of lipoid metabolism', 'prophylactic vaccination and inoculation','Other Supplementary', 'Essential hypertension',
       'Heart failure', 'Hyperkinetic syndrome of childhood',
       'Hyperplasia of prostate', 'Late effects of cerebrovascular disease',
       'Neurotic disorders', 'Nonspecific findings on examination of urine',
       'Osteoarthrosis and allied disorders', 'Other Comorbidity categories',
       'Other and unspecified anemias', 'Other and unspecified arthropathies',
       'Other forms of chronic ischemic heart disease',
       'Other peripheral vascular disease',
       'Symptoms involving nervous and musculoskeletal systems',
       'Symptoms involving skin and other integumentary tissue','State_AK', 'State_AL', 'State_AR', 'State_AZ',
       'State_CA', 'State_CO', 'State_CT', 'State_DC', 'State_DE', 'State_FL',
       'State_GA', 'State_HI', 'State_IA', 'State_ID', 'State_IL', 'State_IN',
       'State_KS', 'State_KY', 'State_LA', 'State_MA', 'State_MD', 'State_ME',
       'State_MI', 'State_MN', 'State_MO', 'State_MS', 'State_MT', 'State_NC',
       'State_ND', 'State_NE', 'State_NH', 'State_NJ', 'State_NM', 'State_NV',
       'State_NY', 'State_OH', 'State_OK', 'State_OR', 'State_PA', 'State_PR',
       'State_SC', 'State_SD', 'State_TN', 'State_TX', 'State_UT', 'State_VA',
       'State_VT', 'State_WA', 'State_WV', 'State_WY']

Y_cols = ["DMIndicator"]

train = data.iloc[train.index,:]
test = data.iloc[test.index,:]

X_train = data[X_cols].iloc[train.index,:]
y_train = data[Y_cols].iloc[train.index,:]
X_test = data[X_cols].iloc[test.index,:]


# In[97]:


plt.figure(figsize=(20,8));
corr_heatmap = sns.heatmap(data[Y_cols+X_cols[:33]].corr(),vmin=-1, vmax=1,cmap='BrBG',square=True);
corr_heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=10);


# ### Oversampling

# In[98]:


cnt_non_diabete = train[train['DMIndicator'] == 0]["DMIndicator"].count()
train_class_diabete = train[train['DMIndicator'] == 1]
train_class_non_diabete = train[train['DMIndicator'] == 0]

#OverSampling
train_class_diabete_oversample = train_class_diabete.sample(cnt_non_diabete, replace=True)
train_oversampled = pd.concat([train_class_non_diabete, train_class_diabete_oversample], axis=0)

print('Random over-sampling:')
print(train_oversampled['DMIndicator'].value_counts())

X_train_oversampled = train_oversampled[X_cols]
y_train_oversampled = train_oversampled[Y_cols]


# ### Standardization

# In[99]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train_oversampled_std = scalar.fit_transform(X_train_oversampled)
X_test_std = scalar.fit_transform(X_test)


# ## Modeling

# ### logistic regression

# In[100]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(penalty='l2',random_state=42)
logreg.fit(X_train_oversampled_std, y_train_oversampled)
y_train_pred = logreg.predict(X_train_oversampled_std)
y_test_pred = logreg.predict(X_test_std)


# ### Cross validation 

# In[102]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(logreg, X_train_oversampled_std, y_train_oversampled, cv=5)
print('Cross-Validation Accuracy Scores', scores)


# In[103]:


from collections import Counter
Counter(y_train_pred)


# ### Classification report 

# In[104]:


from sklearn.metrics import accuracy_score, classification_report
print(classification_report(y_train_oversampled, y_train_pred))


# ### Confusion matrix 

# In[105]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train_oversampled, y_train_pred)
sns.heatmap(cm, annot=True, fmt = 'd', cmap = 'Blues', annot_kws = {'size': 16})
plt.xlabel('Predicted')
plt.ylabel('Actual');


# ### ROC curve

# In[106]:


# create a ROC curve
from sklearn import metrics

y_pred_proba = logreg.predict_proba(X_train_oversampled_std)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_train_oversampled,y_pred_proba)

auc = metrics.roc_auc_score(y_train_oversampled,y_pred_proba)
plt.plot(fpr,tpr,label = "auc="+str(auc))
plt.legend(loc=4)
plt.title("ROC curve")
plt.show()


# ### Feature coefs 

# In[109]:


data_feature = pd.DataFrame({"feature":X_train_oversampled.columns,"coefs":logreg.coef_[0]}).sort_values("coefs",ascending=False)
sns.set(rc = {'figure.figsize':(6,15)})
sns.barplot(x="coefs",y="feature",data=data_feature,palette="husl");
plt.ylabel("");


# ### Brier score 

# In[159]:


from sklearn.metrics import brier_score_loss
brier_score_loss(y_train_oversampled, y_pred_proba)


# ### Output forecast DMIndicatorForecast

# In[212]:


test_SyncPatientForecast = pd.DataFrame({"PracticeGuid":test["PracticeGuid"],"DMIndicatorForecast":y_test_pred})
test_SyncPatientForecast.to_csv("test_SyncPatientForecast.csv")


# In[213]:


test_SyncPatientForecast

