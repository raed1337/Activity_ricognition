from sklearn.linear_model import OrthogonalMatchingPursuit, RANSACRegressor, LogisticRegression, ElasticNetCV, \
    HuberRegressor, Ridge, Lasso, LassoCV, Lars, BayesianRidge, SGDClassifier, LogisticRegressionCV, RidgeClassifier
#Here importing a true army of different models
from sklearn.preprocessing import MinMaxScaler
#Give this guy an interval and he scales your data to fit inside the interval
import seaborn as sns
#This guy draws plots nicely
import numpy as np 
#This guy does math, linear algebra
import pandas as pd 
#This guy is Mr Data Frames
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#This guy draws plots too. But not so nicely
from sklearn.preprocessing import LabelEncoder


def importDataSegemntation():
    df_data = pd.read_csv("./ConfLongDemo_JSI.csv",header=None)
    return df_data
#We import the data we got after clustering our tables

train=importDataSegemntation()
train.columns=['Sequence Name','tagID','Time stamp','Date','x coordinate','y coordinate','z coordinate','activity']
#We gather the names of our columns

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values))
        train[c] = lbl.transform(list(train[c].values))


def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted) - np.log1p(y_real), 2)))
#Remember the mean square error? MSE? Well, there's his little brother, the Mean Square Logarithmic Error. MSLE
#Also, meet their child: the Root Mean Square Logarithmic Error.
#RMSLE measures the ratio between actual and predicted.
#It can be used when you don’t want to penalize huge differences when both the values are huge numbers.
#Also, this can be used when you want to penalize under estimates more than over estimates.

def procenterror(y_predicted, y_real):
    return np.round(np.mean(np.abs(y_predicted - y_real)) / np.mean(y_real) * 100, 1)
#Average percentage of error

from sklearn.linear_model import OrthogonalMatchingPursuit, RANSACRegressor, LogisticRegression, ElasticNetCV, \
    HuberRegressor, Ridge, Lasso, LassoCV, Lars, BayesianRidge, SGDClassifier, LogisticRegressionCV, RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
    recall_score

#I'm thinking of reimporting all libraries to make sure I didn't forget any

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
#These are the parameters of every single model we have in our list. For SOME REASON, we only used the Decision Tree
#There are over 99 good models in history and decision tree ain't one
#and we picked decision tree.

n_col = 36
print("Correlation between variables")
print(train.corr())
X = train.drop(['activity'], axis=1)
X = X.drop(['Time stamp'], axis=1)
X = X.drop(['tagID'], axis=1)
X = X.drop(['Sequence Name'], axis=1)


Y = train['activity']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=4)


# X=X.fillna(value=0)
# scaler = MinMaxScaler()
# scaler.fit(X)
# X=scaler.transform(X)
# poly = PolynomialFeatures(2)
# X=poly.fit_transform(X)

names = [
    'DecisionTree',
    'RandomForestClassifier',
    'LDA',
   # 'ElasticNet',
    #'kSVC',
   # 'GaussianNB',
    'KNN',
    #'SVC',
    #'GridSearchCV',
    #'HuberRegressor',
   # 'Ridge',
   # 'Lasso',
   # 'LassoCV',
   # 'Lars',
   # 'BayesianRidge',
    # 'SGDClassifier',
    #'RidgeClassifier',
    'LogisticRegression',
    #'OrthogonalMatchingPursuit',
    # 'RANSACRegressor',
]

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

classifiers = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LinearDiscriminantAnalysis(),
    #ElasticNetCV(cv=10, random_state=0),
    #SVC(),
    #GaussianNB(),
    KNeighborsClassifier(n_neighbors=3),
    #SVC(kernel='rbf'),
    #GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),
    #HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100, epsilon=2.95),
    #Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),
    #Lasso(alpha=0.05),
    #LassoCV(),
    #Lars(n_nonzero_coefs=10),
    #BayesianRidge(),
    #SGDClassifier(),
    #RidgeClassifier(),
    LogisticRegression(),
    # #BINAIRE
    #OrthogonalMatchingPursuit(),
    #RANSACRegressor(),
]
models = zip(names, classifiers)
ResultAccuracy = []
CV_Score=[]

#Basically we're gonna loop over our models and execute the classifier then we're going to view the classification report
#and pick the best model. Since we picked on decision tree (for reasons beyond me) we only view one model
#Yay
from sklearn import model_selection
seed=7
#Seed a partir du quel on va faire des operation aléatoires. Pour maintenir les datasets sont kifkif durant toute les itérations
# Cross validation 10 data set .
scoring = 'accuracy'

print("TRAINIGNG STARTED ...")
for name, clf in models:
    print('model is ',name)
    regr = clf.fit(X_train, y_train)
    # print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)
    print(name, '%error', procenterror(regr.predict(X_test), y_test), 'rmsle', rmsle(regr.predict(X_test), y_test))

    # Confusion Matrix
    """print(name, 'Confusion Matrix')
    conf = confusion_matrix(y_test, np.round(regr.predict(X_test)))
    label = np.sort(y_test.unique())
    print(conf)"""
    print(name, 'Confusion Matrix')
    conf = confusion_matrix(Y, np.round(regr.predict(X)))
    label = np.sort(Y.unique())
    sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label, cmap="YlGnBu")
    plt.show()

    # Classification Report
    print(name, 'Classification Report')
    classif = classification_report(y_test, np.round(regr.predict(X_test)))
    print(classif)

    # Accuracy
    print('--' * 40)
    logreg_accuracy = round(accuracy_score(y_test, np.round(regr.predict(X_test))) * 100, 2)
    print('Accuracy', logreg_accuracy, '%')
    ResultAccuracy.append(logreg_accuracy)
    # Save accuracy of each model into ResultAccuracy to compare them later
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(clf, X, Y, cv=kfold, scoring=scoring)
    CV_Score.append(cv_results)


print(ResultAccuracy)
print(names)
AccuracyNames = pd.DataFrame(ResultAccuracy, index=names, columns=['accuracy'])
AccuracyNames.sort_values(by='accuracy').plot(kind='bar')
plt.show()


fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(CV_Score)
ax.set_xticklabels(names)
plt.show()