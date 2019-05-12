#####               I am lazy                               #####
#####           Never mind^_^                               #####
#####            STACKING                                   #####
#####    I know it is the most important of this part       #####
#####     so I put it at the first                          #####      
#####     use f1 as scoring                                 #####





from sklearn.model_selection import KFold
mods=[SVC(class_weight='balanced',
          C=0.01),
      LogisticRegression(class_weight='balanced',
                         C=0.01),
      tree.DecisionTreeClassifier(criterion="gini",
                                  min_samples_leaf=5,
                                  max_depth=6,
                                  class_weight='balanced'),
      RandomForestClassifier(n_estimators=100,
                             max_depth=6,
                             min_samples_split=2,
                             class_weight="balanced"),
      ExtraTreesClassifier(n_estimators=100,
                           max_depth=6,
                           min_samples_split=5,
                           class_weight="balanced"),
      XGBClassifier(learning_rate=0.01,
                    n_estimators=180,
                    max_depth=6,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    scale_pos_weight=70),
      BernoulliNB(alpha=1.0,
                  binarize=0.0,
                  class_prior=[0.99,0.1],
                  fit_prior=True)]
x_train,x_test,y_train,y_test = train_test_split(x2,
                                                 y2,
                                                 test_size=0.3)
kf=KFold(n_splits=5,shuffle=True)
train_sets=pd.DataFrame(index=x_train.index,columns=['0','1','2','3','4','5','6'])
test_sets=pd.DataFrame(index=x_test.index,columns=['0','1','2','3','4','5','6'])
j=0
for mmm in mods:
    second_level_train_set=pd.DataFrame(index=x_train.index,columns=['fake'])
    second_level_test_set=pd.DataFrame(index=x_test.index,columns=['fake'])
    test_nfolds_sets=pd.DataFrame(index=x_test.index,columns=['0','1','2','3','4'])
    for i,(train_index,test_index) in enumerate(kf.split(x_train)):
        x_tra,y_tra=x_train.iloc[train_index],y_train.iloc[train_index]
        x_tst,y_tst=x_train.iloc[test_index],y_train.iloc[test_index]
        mmm.fit(x_tra,y_tra)
        second_level_train_set['fake'].iloc[test_index]=mmm.predict(x_tst)
        test_nfolds_sets[str(i)]=mmm.predict(x_test)
        second_level_test_set['fake']=test_nfolds_sets.mean(axis=1)
    train_sets[str(j)]=second_level_train_set['fake']
    test_sets[str(j)]=second_level_test_set['fake']
    j=j+1
xgbx=XGBClassifier(learning_rate=0.01,
                  n_estimators=150,
                  max_depth=6,
                  min_child_weight=1,
                  gamma=0,
                  subsample=0.8,
                  colsample_bytree=0.8,
                   objective='binary:logistic',
                  scale_pos_weight=70)
xgbx.fit(train_sets,y_train)
rrr=xgbx.predict(test_sets)
ss1=y_test[rrr==1]['fake'].value_counts()[1]
ss2=pd.Series(rrr).value_counts()[1]
ss3=y_test['fake'].value_counts()[1]
ff1=ss1/ss2
ff2=ss1/ss3
ff3=2*ff1*ff2/(ff1+ff2)
print(ff1)
print(ff2)
print(ff3)







#####content:
#####         svm,lr,tree,mlp,randomforest,extraforest,BernoulliNB,gbdt,adaboost,xgb,voting,bagging









import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#######data pre#####
train=pd.read_csv('data/FT_Camp_5/Train.csv',encoding='gbk',engine='python')
xt=pd.read_csv('data/FT_Camp_5/X_test.csv',encoding='gbk',engine='python')
x=train[train.columns[0:35]]
y=train[['stockcode','fake']]
x=x.set_index('stockcode')
y=y.set_index('stockcode')
xt=xt.set_index('stockcode')
ax=x[y['fake']==0]
nx=x[y['fake']==1]
ax1=ax.fillna(ax.mean())
nx1=nx.fillna(ax.mean())
xt1=xt.fillna(ax.mean())
for i in ax1.columns:
    if i!='Opinion' and i!='Pre_np' and i!='Neg_Dednp_times':
        ax1[i]=(ax1[i]-ax[i].mean())/ax[i].std()
        nx1[i]=(nx1[i]-ax[i].mean())/ax[i].std()
        xt1[i]=(xt1[i]-ax[i].mean())/ax[i].std()
ay=y[y['fake']==0]
ny=y[y['fake']==1]
x1=ax1
y1=ay
for i in range(70):
    x1=pd.concat([x1,nx1])
    y1=pd.concat([y1,ny])
x2=pd.concat([ax1,nx1])
y2=pd.concat([ay,ny])
ax1.corr()
col=ax.columns



#####    oneclasssvm
from sklearn import svm
clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(ax1)
r4=clf.predict(xt1)
pd.Series(r4).value_counts()

#####      svc
from sklearn.svm import SVC
svc = SVC(class_weight='balanced')
svc.fit(x[['Opinion','Pre_np','OtherRec_to_Cur','Hold_Top5','Reve_growth_rate','Mainb_gp_ratio','AR_to_Reve','Sale_grop_ratio','Pre_to_Cur','Mainb_pro_ratio','Inventory_to_Cur']],y1)
r4=svc.predict(xt1[['Opinion','Pre_np','OtherRec_to_Cur','Hold_Top5','Reve_growth_rate','Mainb_gp_ratio','AR_to_Reve','Sale_grop_ratio','Pre_to_Cur','Mainb_pro_ratio','Inventory_to_Cur']])
pd.Series(r4).value_counts()
r4=svc.predict(xt1[['Opinion','Pre_np','OtherRec_to_Cur','Hold_Top5','Reve_growth_rate','Mainb_gp_ratio','AR_to_Reve','Sale_grop_ratio','Pre_to_Cur','Mainb_pro_ratio','Inventory_to_Cur']])
r5=svc.predict(ax1[['Opinion','Pre_np','OtherRec_to_Cur','Hold_Top5','Reve_growth_rate','Mainb_gp_ratio','AR_to_Reve','Sale_grop_ratio','Pre_to_Cur','Mainb_pro_ratio','Inventory_to_Cur']])
r6=svc.predict(nx1[['Opinion','Pre_np','OtherRec_to_Cur','Hold_Top5','Reve_growth_rate','Mainb_gp_ratio','AR_to_Reve','Sale_grop_ratio','Pre_to_Cur','Mainb_pro_ratio','Inventory_to_Cur']])
r7=svc.predict(x1[['Opinion','Pre_np','OtherRec_to_Cur','Hold_Top5','Reve_growth_rate','Mainb_gp_ratio','AR_to_Reve','Sale_grop_ratio','Pre_to_Cur','Mainb_pro_ratio','Inventory_to_Cur']])
print(pd.Series(r4).value_counts())
print(pd.Series(r5).value_counts(1)[0])
print(pd.Series(r6).value_counts(1)[1])
print(pd.Series(r7).value_counts())
f1=pd.Series(r6).value_counts(1)[1]
f2=pd.Series(r6).value_counts()[1]/(pd.Series(r6).value_counts()[1]+pd.Series(r5).value_counts()[1])
print(f1)
print(f2)
print(f1*f2/(f1+f2)*2)
from sklearn.svm import SVC
svc1 = SVC(class_weight='balanced',C=0.01)
x21=x2[['Opinion','Pre_np','OtherRec_to_Cur','Hold_Top5',
        'Reve_growth_rate','Mainb_gp_ratio','AR_to_Reve',
        'Sale_grop_ratio','Pre_to_Cur','Mainb_pro_ratio','Inventory_to_Cur']]
mff1=0
mff2=0
mff3=0
for i in range(5):
    x_train,x_test,y_train,y_test = train_test_split(x21,
                                                     y2,
                                                     test_size=0.3)
    svc1.fit(x_train,y_train)
    r42=svc1.predict(x_test)
    ss1=y_test[r42==1]['fake'].value_counts()[1]
    ss2=pd.Series(r42).value_counts()[1]
    ss3=y_test['fake'].value_counts()[1]
    ff1=ss1/ss2
    ff2=ss1/ss3
    ff3=2*ff1*ff2/(ff1+ff2)
    mff1=mff1+ff1
    mff2=mff2+ff2
    mff3=mff3+ff3
print(mff1/5)
print(mff2/5)
print(mff3/5)



#####      lr
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x1[['Opinion','Pre_np','OtherRec_to_Cur','Hold_Top5','Reve_growth_rate','Mainb_gp_ratio','AR_to_Reve','Sale_grop_ratio','Pre_to_Cur','Mainb_pro_ratio','Inventory_to_Cur']],y1)
r4=lr.predict(xt1[['Opinion','Pre_np','OtherRec_to_Cur','Hold_Top5','Reve_growth_rate','Mainb_gp_ratio','AR_to_Reve','Sale_grop_ratio','Pre_to_Cur','Mainb_pro_ratio','Inventory_to_Cur']])
pd.Series(r4).value_counts()
from sklearn.linear_model import LogisticRegression
lr1=LogisticRegression(class_weight='balanced',C=0.01)
x21=x2[['Opinion','Pre_np','OtherRec_to_Cur','Hold_Top5',
        'Reve_growth_rate','Mainb_gp_ratio','AR_to_Reve',
        'Sale_grop_ratio','Pre_to_Cur','Mainb_pro_ratio','Inventory_to_Cur']]
mff1=0
mff2=0
mff3=0
for i in range(5):
    x_train,x_test,y_train,y_test = train_test_split(x21,
                                                     y2,
                                                     test_size=0.3)
    lr1.fit(x_train,y_train)
    r42=lr1.predict(x_test)
    ss1=y_test[r42==1]['fake'].value_counts()[1]
    ss2=pd.Series(r42).value_counts()[1]
    ss3=y_test['fake'].value_counts()[1]
    ff1=ss1/ss2
    ff2=ss1/ss3
    ff3=2*ff1*ff2/(ff1+ff2)
    mff1=mff1+ff1
    mff2=mff2+ff2
    mff3=mff3+ff3
print(mff1/5)
print(mff2/5)
print(mff3/5)



####     tree
from sklearn import tree
tr1=tree.DecisionTreeClassifier(criterion="gini",
                                min_samples_leaf=5,
                                max_depth=6,
                                class_weight='balanced'
                               )
x21=x2[['Opinion','Pre_np','OtherRec_to_Cur','Hold_Top5',
        'Reve_growth_rate','Mainb_gp_ratio','AR_to_Reve',
        'Sale_grop_ratio','Pre_to_Cur','Mainb_pro_ratio','Inventory_to_Cur']]
mff1=0
mff2=0
mff3=0
for i in range(5):
    x_train,x_test,y_train,y_test = train_test_split(x21,
                                                     y2,
                                                     test_size=0.3)
    tr1.fit(x_train,y_train)
    r42=tr1.predict(x_test)
    ss1=y_test[r42==1]['fake'].value_counts()[1]
    ss2=pd.Series(r42).value_counts()[1]
    ss3=y_test['fake'].value_counts()[1]
    ff1=ss1/ss2
    ff2=ss1/ss3
    ff3=2*ff1*ff2/(ff1+ff2)
    mff1=mff1+ff1
    mff2=mff2+ff2
    mff3=mff3+ff3
print(mff1/5)
print(ff2/5)
print(mff3/5)


#####       mlp
mlp1 = MLPClassifier(solver='sgd',
                     activation='logistic',
                     alpha=1e-4,
                     hidden_layer_sizes=(50,50,50),
                     max_iter=10,
                     shuffle=True,
                     verbose=10,
                     learning_rate_init=.1,
                     scale_pos_weight=1) 
x_train,x_test,y_train,y_test = train_test_split(x1,
                                                 y1,
                                                 test_size=0.3)
mlp1.fit(x_train,y_train)
x_test=x_test.reset_index()
x_test=x_test.drop_duplicates()
x_test=x_test.set_index('stockcode')
y_test=y_test.reset_index()
y_test=y_test.drop_duplicates()
y_test=y_test.set_index('stockcode')
r42=mlp1.predict(x_test)
ss1=y_test[r42==1]['fake'].value_counts()[1]
ss2=pd.Series(r42).value_counts()[1]
ss3=y_test['fake'].value_counts()[1]
ff1=ss1/ss2
ff2=ss1/ss3
print(ff1)
print(ff2)
print(2*ff1*ff2/(ff1+ff2))

#####randomforest
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,
                          max_depth=6,
                          min_samples_split=2,
                         random_state=0)
rf.fit(x1,y1)
r4=rf.predict(xt1)
r5=rf.predict(ax1)
r6=rf.predict(nx1)
r7=rf.predict(x1)
print(pd.Series(r4).value_counts())
print(pd.Series(r5).value_counts(1)[0])
print(pd.Series(r6).value_counts(1)[1])
print(pd.Series(r7).value_counts())
f1=pd.Series(r6).value_counts(1)[1]
f2=pd.Series(r6).value_counts()[1]/(pd.Series(r6).value_counts()[1]+pd.Series(r5).value_counts()[1])
print(f1)
print(f2)
print(f1*f2/(f1+f2)*2)
rf3=RandomForestClassifier(n_estimators=100,
                           max_depth=6,
                           min_samples_split=2,
                           class_weight="balanced")


######extratrees
from sklearn.ensemble import ExtraTreesClassifier
etc=ExtraTreesClassifier(n_estimators=100, max_depth=10,min_samples_split=2, random_state=0)
etc.fit(x1,y1)
r4=etc.predict(xt1)
r5=etc.predict(ax1)
r6=etc.predict(nx1)
r7=etc.predict(x1)
print(pd.Series(r4).value_counts())
print(pd.Series(r5).value_counts(1)[0])
print(pd.Series(r6).value_counts(1)[1])
print(pd.Series(r7).value_counts())
f1=pd.Series(r6).value_counts(1)[1]
f2=pd.Series(r6).value_counts()[1]/(pd.Series(r6).value_counts()[1]+pd.Series(r5).value_counts()[1])
print(f1)
print(f2)
print(f1*f2/(f1+f2)*2)


#####adaboost
from sklearn.ensemble import AdaBoostClassifier
adab = AdaBoostClassifier(n_estimators=1000)
adab.fit(x1,y1)
r4=adab.predict(xt1)
r5=adab.predict(ax1)
r6=adab.predict(nx1)
r7=adab.predict(x1)
print(pd.Series(r4).value_counts())
print(pd.Series(r5).value_counts(1)[0])
print(pd.Series(r6).value_counts(1)[1])
print(pd.Series(r7).value_counts())
f1=pd.Series(r6).value_counts(1)[1]
f2=pd.Series(r6).value_counts()[1]/(pd.Series(r6).value_counts()[1]+pd.Series(r5).value_counts()[1])
print(f1)
print(f2)
print(f1*f2/(f1+f2)*2)


#####gradientboost
from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier(max_depth=6,
                              min_samples_split=2,
                              n_estimators=50,learning_rate=0.05,subsample=0.8)
gb.fit(x1,y1)
r4=gb.predict(xt1)
r5=gb.predict(ax1)
r6=gb.predict(nx1)
r7=gb.predict(x1)
print(pd.Series(r4).value_counts())
print(pd.Series(r5).value_counts(1)[0])
print(pd.Series(r6).value_counts(1)[1])
print(pd.Series(r7).value_counts())
f1=pd.Series(r6).value_counts(1)[1]
f2=pd.Series(r6).value_counts()[1]/(pd.Series(r6).value_counts()[1]+pd.Series(r5).value_counts()[1])
print(f1)
print(f2)
print(f1*f2/(f1+f2)*2)


#####xgb
from xgboost import XGBClassifier
xgb=XGBClassifier(learning_rate=0.01,
                  n_estimators=150,
                  max_depth=6,
                  min_child_weight=1,
                  gamma=0,
                  subsample=0.8,
                  colsample_bytree=0.8,
                   objective='binary:logistic',
                  scale_pos_weight=1)
xgb.fit(x1,y1)
r4=xgb.predict(xt1)
r5=xgb.predict(ax1)
r6=xgb.predict(nx1)
r7=xgb.predict(x1)
print(pd.Series(r4).value_counts())
print(pd.Series(r5).value_counts(1)[0])
print(pd.Series(r6).value_counts(1)[1])
print(pd.Series(r7).value_counts())
f1=pd.Series(r6).value_counts(1)[1]
f2=pd.Series(r6).value_counts()[1]/(pd.Series(r6).value_counts()[1]+pd.Series(r5).value_counts()[1])
print(f1)
print(f2)
print(f1*f2/(f1+f2)*2)



#####gridsearch
from sklearn.model_selection import GridSearchCV
xgb3=XGBClassifier(objective='binary:logistic',
                   scale_pos_weight=70)
par={'learning_rate':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09],
     'n_estimators':[25,50,75,100,125,150,175,200,225,250],
     'max_depth':[1,2,3,4,5,6,7,8,9,10],
     'min_child_weight':[1],
     'gamma':[0],
     'subsample':[0.8],
     'colsample_bytree':[0.8]}
xgbx=GridSearchCV(xgb3,par,scoring='f1')
xgbx.fit(x2,y2)



#####voting
from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('1',gbdt), ('2',b1), ('3', adab)], voting='hard')
eclf.fit(x1,y1)
r4=eclf.predict(xt1)
r5=eclf.predict(ax1)
r6=eclf.predict(nx1)
r7=eclf.predict(x1)
print(pd.Series(r4).value_counts())
print(pd.Series(r5).value_counts(1)[0])
print(pd.Series(r6).value_counts(1)[1])
print(pd.Series(r7).value_counts())
f1=pd.Series(r6).value_counts(1)[1]
f2=pd.Series(r6).value_counts()[1]/(pd.Series(r6).value_counts()[1]+pd.Series(r5).value_counts()[1])
print(f1)
print(f2)
print(f1*f2/(f1+f2)*2)


#####bagging
from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(adab,max_samples=0.5, max_features=0.5)
bag.fit(x1,y1)
r4=bag.predict(xt1)
r5=bag.predict(ax1)
r6=bag.predict(nx1)
r7=bag.predict(x1)
print(pd.Series(r4).value_counts())
print(pd.Series(r5).value_counts(1)[0])
print(pd.Series(r6).value_counts(1)[1])
print(pd.Series(r7).value_counts())
f1=pd.Series(r6).value_counts(1)[1]
f2=pd.Series(r6).value_counts()[1]/(pd.Series(r6).value_counts()[1]+pd.Series(r5).value_counts()[1])
print(f1)
print(f2)
print(f1*f2/(f1+f2)*2)





####  BernoulliNB
from sklearn.naive_bayes import BernoulliNB
nnb=BernoulliNB()
nnb.fit(x1,y1)


