#import packages which are necessary
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, skewtest
import matplotlib

train = 'C:/Materials/Classes/Big Data/House Prices/Data/train.csv'
test = 'C:/Materials/Classes/Big Data/House Prices/Data/test.csv'
df_train = pd.read_csv(train)
df_test = pd.read_csv(test)
df_full =df_train.append(df_test)


fullCat = df_full.select_dtypes(include=['object']).index
fullCont = df_full.dtypes[df_full.dtypes !='object'].index


#Skewing
skewed_data = df_full[fullCont].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_data = skewed_data[skewed_data > 0.75]
skewed_data = skewed_data.index
df_full[skewed_data] = np.log(df_full[skewed_data] + 1)

#getting dummies
df_full = pd.get_dummies(df_full)

df_full = df_full.fillna(df_full[:df_train.shape[0]].mean())


def heatmap(df,labels):
    cm = np.corrcoef(df[labels].dropna().values.T)
    sns.set(font_scale=1)
    hm = sns.heatmap(cm,
                     cbar= False,
                     annot =False,
                     square= True,
                     vmax=1
                     )

    #heatmap = ax.pcolor(nba_sort, cmap=plt.cm.Blues, alpha=0.8
    color_map = plt.cm.Blues
    plt.pcolor(cm,cmap=color_map)
    plt.colorbar().set_label("Features", rotation=270)
    hm.set_xticklabels(labels, rotation=90)
    hm.set_yticklabels(labels[::-1], rotation=0)

    return hm,cm


Features = list(df_full[fullCont].columns.values)

plt.figure(figsize = (20,10))
htmp,corrm = heatmap(df_full[fullCont],Features)
plt.show()


#Feature Selection
df_train['SalePrice'] = np.log(df_train['SalePrice'])
del df_full['SalePrice']
trainData = df_full[:df_train.shape[0]]
testData = df_full[:df_test.shape[0]]
YData = df_train.SalePrice


#Applying Lasso Model

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression
def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, trainData, YData, scoring="neg_mean_squared_error", cv=5))
    return(rmse)

model_lasso = LassoCV(alphas = [1, 0.1, 0.001,0.0005], selection='random', max_iter=15000).fit(trainData, YData)
res = rmse_cv(model_lasso)
#print(res)
print("Lasso Mean:",res.mean())
print("Lasso Min: ",res.min())



coef = pd.Series(model_lasso.coef_, index = trainData.columns)
#print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh",color = 'r')
plt.title("Coefficients used in the Lasso Model")

plt.show()



# matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)
# preds = pd.DataFrame({"preds":model_lasso.predict(trainData), "true":YData})
# preds["residuals"] = preds["true"] - preds["preds"]
# preds.plot(x = "preds", y = "residuals",kind = "scatter")
# plt.show()


test_preds = np.expm1(model_lasso.predict(testData))
result = pd.DataFrame()
result['Id'] = df_test['Id']
result["SalePrice"] = test_preds
result.to_csv("C:/Materials/Classes/Big Data/House Prices/Data/lasso.csv", index=False)

#Applying Linear Regression

from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression = linear_regression.fit(trainData,YData)
res1 = rmse_cv(linear_regression)
test_preds = (linear_regression.predict(testData))
submission2 = pd.DataFrame()
submission2['Id'] = df_test['Id']
submission2["SalePrice"] = np.exp(test_preds)
submission2.to_csv("C:/Materials/Classes/Big Data/House Prices/Data/linear.csv", index=False)


print("Linear Regression Mean",res1.mean())
print("Linear Regression Min:",res1.min())



#Logistic Regression
from sklearn.linear_model import LogisticRegression


#from sklearn.cross_validation import train_test_split
#X = df_full[:df_train.shape[0]].values
#y = df_train.SalePrice.values
#X_train, X_test, y_train, y_test = \
#train_test_split(X, y, test_size=0.3, random_state=0)
#from sklearn.preprocessing import StandardScaler
#stdsc = StandardScaler()
#X_train_std = stdsc.fit_transform(X_train)
#X_test_std = stdsc.transform(X_test)
#y_train = y_train.astype(int)

#y_test = y_test.astype(int)


#lr = LogisticRegression(penalty='l1', C=0.1)
#lr.fit(trainData,YData)
#y_test = testData.SalePrice
#y_pred = lr.predict(X_test_std)
#rms = np.sqrt(mean_squared_error(y_test, y_pred))
#print(rms)

#print('Training accuracy:', lr.score(X_train_std, y_train))
#print('Test accuracy:', lr.score(X_test_std, y_test))



# fig = plt.figure()
# ax = plt.subplot(111)
#
# colors = ['blue', 'green', 'red', 'cyan',
#           'magenta', 'yellow', 'black',
#           'pink', 'lightgreen', 'lightblue',
#           'gray', 'indigo', 'orange']
#
# weights, params = [], []
# for c in np.arange(-4, 6):
#     lr = LogisticRegression(penalty='l1', C=10 ** c, random_state=0)
#     lr.fit(trainData, YData)
#     weights.append(lr.coef_[1])
#     params.append(10 ** c)
#
# weights = np.array(weights)
#
# for column, color in zip(range(weights.shape[1]), colors):
#     plt.plot(params, weights[:, column],
#              label=df_full.columns[column + 1],
#              color=color)
# plt.axhline(0, color='black', linestyle='--', linewidth=3)
# plt.xlim([10 ** (-5), 10 ** 5])
# plt.ylabel('weight coefficient')
# plt.xlabel('C')
# plt.xscale('log')
# plt.legend(loc='upper left')
# ax.legend(loc='upper center',
#           bbox_to_anchor=(1.38, 1.03),
#           ncol=1, fancybox=True)
# plt.show()



#Applying XGBoost Model

import xgboost as xgb

regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200)
regr.fit(trainData,YData)

print("XGBoost Mean ", rmse_cv(regr).mean())
res2 = rmse_cv(regr).mean()
pred_xgb=  regr.predict(testData)

#xgbCoef = pd.Series(pred_xgb.feature_importances_, index = trainData.columns)
#print("XGB picked " + str(sum(xgbCoef != 0)) + " variables and eliminated the other " +  str(sum(xgbCoef == 0)) + " variables")
#imp_coef1 = pd.concat([xgbCoef.sort_values().head(10),
 #                    xgbCoef.sort_values().tail(10)])
#matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
#imp_coef1.plot(kind = "rbarh")
#plt.title("Coefficients in the XGB Model")
#plt.show()

#importance = .get_fscore(fmap='xgb.fmap')
#importance = sorted(importance.items())
#df = pd.DataFrame(importance, columns=['feature', 'fscore'])
#df['fscore'] = df['fscore'] / df['fscore'].sum()

#plt.figure()
#df.plot()
#df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
#plt.title('XGBoost Feature Importance')
#plt.xlabel('relative importance')


submission1 = pd.DataFrame()
submission1['Id'] = df_test['Id']
submission1["SalePrice"] = np.exp(pred_xgb)
submission1.to_csv("C:/Materials/Classes/Big Data/House Prices/Data/xgb.csv", index=False)



# Run prediction on the Kaggle test set.
#y_pred_xgb = regr.predict(test_df_munged)



# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# #x_train,x_val,y_train,y_val = train_test_split(df_full.iloc[:,0:310],df_full[311],random_state = 0)
# X_train, X_test, y_train, y_test = \
# train_test_split(X, y, test_size=0.3, random_state=0)
#
# # Fill the missing values of numeric data with the mean of the columns
# x_train = X_train.fillna(X_train.mean())
# x_val = X_test.fillna(X_test.mean())
#
# # Make sure that shape of the dataframe is not wrong
# print(x_train.shape)
# print(x_val.shape)
# print(y_train.shape)
# print(y_test.shape)
labels = ['LinearRegression','Laaso Model','XGBoost']
#meanScores = [int(min(res)),int(min(res1)),int(min(res2))]
meanScores = [0.1655,0.1229,0.119]
#print(meanScores)
#fig = plt.figure()
ind_scrs = np.arange(0,len(meanScores))
width = 0.3
fig, ax = plt.subplots()
rects = ax.bar(ind_scrs, meanScores, width, color = 'b')
ax.set_xticks(np.array(ind_scrs) + width/2)
ax.set_xticklabels(labels)
ax.set_ylabel('RMSE on test set')
ax.set_ylim([0,0.2])
plt.show()



