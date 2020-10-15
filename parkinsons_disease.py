import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

parkinsons=pd.read_csv('parkinsons.data')
print(parkinsons.columns)
y=parkinsons.loc[:,'status'].values
features=parkinsons.loc[:,parkinsons.columns!='status'].values[:,1:]
scaler=MinMaxScaler((-1,1))
X=scaler.fit_transform(features)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=7)
xgb=XGBClassifier()
xgb.fit(X_train, y_train)
y_pred=xgb.predict(X_test)
print('accuracy score= ', accuracy_score(y_test, y_pred)*100)

