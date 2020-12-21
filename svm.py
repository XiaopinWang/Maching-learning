from sklearn.model_selection import validation_curve
from sklearn import svm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
diabets_df = pd.read_csv('diabetes_data_upload.csv')
X = diabets_df.drop(['class'], axis = 1)
Y = diabets_df['class']
X = pd.get_dummies(X, prefix_sep='_')
Y = preprocessing.LabelEncoder().fit_transform(Y)
X2 = preprocessing.StandardScaler().fit_transform(X)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X2, Y, test_size = 0.3)
Numeric_df = pd.DataFrame(X)
Numeric_df['class'] = Y
corr = Numeric_df.corr()
corr_y = abs(corr["class"])
highest_corr = corr_y[corr_y > 0.3]
print(highest_corr)
plt.figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
corr2 = Numeric_df[['Gender_Female' , 'Gender_Male' , 'Polyuria_No' , 'Polyuria_Yes' ,  'Polydipsia_No' , 'Polydipsia_Yes' , 'sudden weight loss_No' ,  'sudden weight loss_Yes' , 'Polyphagia_No','Polyphagia_Yes','partial paresis_No','partial paresis_Yes']].corr()
sns.heatmap(corr2, annot=True)
plt.show()
X_Reduced2 = Numeric_df[['Gender_Female' , 'Gender_Male' , 'Polyuria_No' , 'Polyuria_Yes' ,  'Polydipsia_No' , 'Polydipsia_Yes' , 'sudden weight loss_No' ,  'sudden weight loss_Yes' , 'Polyphagia_No','Polyphagia_Yes','partial paresis_No','partial paresis_Yes']]
X_Reduced2 = preprocessing.StandardScaler().fit_transform(X_Reduced2)
#  用于寻找模型中的参数gamma
param_range = np.logspace(-6,-2.3,10)
train_loss,test_loss=validation_curve(SVC(),X_Reduced2,Y,param_name="gamma",param_range=param_range,cv=10, scoring="neg_mean_squared_error")
train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)
plt.plot(param_range,train_loss_mean,"o-",color="r",label="Training")
plt.plot(param_range,test_loss_mean,"o-",color="g",label="Cross-validation")
plt.xlabel("gamma")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()
X_Train3, X_Test3, Y_Train3, Y_Test3 = train_test_split(X_Reduced2, Y, test_size = 0.30)
model = svm.SVC(gamma=0.00042)
model.fit(X_Train3, Y_Train3)
prediction =model.predict(X_Test3)
print(confusion_matrix(Y_Test, prediction))
print(classification_report(Y_Test3, prediction))
print('准确率： ',accuracy_score(prediction,Y_Test3))
