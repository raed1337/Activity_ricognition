
import seaborn as sns
import numpy as np # linear algebra
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import OrthogonalMatchingPursuit,RANSACRegressor,LogisticRegression,ElasticNetCV,HuberRegressor, Ridge, Lasso,LassoCV,Lars,BayesianRidge,SGDClassifier,LogisticRegressionCV,RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler,PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier

def importDataSegemntation():
    df_data = pd.read_csv("./ConfLongDemo_JSI.csv",header=None)
    return df_data

train=importDataSegemntation()
train.columns=['Sequence Name','tagID','Time stamp','Date','x coordinate','y coordinate','z coordinate','activity']

"""train.columns=['Subject id','sequence id','x coordinate','y coordinate','z coordinate','sensor id','activity','instance date']
train['instance date']=pd.to_datetime(train['instance date'],format='%d.%m.%Y  %H:%M:%S:%f')"""

for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values))
        train[c] = lbl.transform(list(train[c].values))


def rmsle(y_predicted, y_real):
    return np.sqrt(np.mean(np.power(np.log1p(y_predicted) - np.log1p(y_real), 2)))


def procenterror(y_predicted, y_real):
    return np.round(np.mean(np.abs(y_predicted - y_real)) / np.mean(y_real) * 100, 1)



from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}

n_col = 36
X = train.drop(['activity'], axis=1)
Y = train['activity']
# X=X.fillna(value=0)
# scaler = MinMaxScaler()
# scaler.fit(X)
# X=scaler.transform(X)
# poly = PolynomialFeatures(2)
# X=poly.fit_transform(X)


names = [
    'DecisionTree',
    #'RandomForestClassifier',
    #'ElasticNet',
    #'SVC',
    #'kSVC',
    'KNN',
    #'GridSearchCV',
    'HuberRegressor',
    'Ridge',
    'Lasso',
    'LassoCV',
    'Lars',
    'BayesianRidge',
    'SGDClassifier',
    'RidgeClassifier',
    'LogisticRegression',
    'OrthogonalMatchingPursuit',
    # 'RANSACRegressor',


]

classifiers = [
    DecisionTreeClassifier( criterion='gini'),
    #RandomForestClassifier(n_estimators=200),
    #ElasticNetCV(cv=10, random_state=0),
    #SVC(),
    #SVC(kernel = 'rbf', random_state = 0),
    KNeighborsClassifier(n_neighbors=3),
    #GridSearchCV(SVC(),param_grid, refit = True, verbose = 1),
    HuberRegressor(fit_intercept=True, alpha=0.0, max_iter=100, epsilon=2.95),
    Ridge(fit_intercept=True, alpha=0.0, random_state=0, normalize=True),
    Lasso(alpha=0.05),
    LassoCV(),
    Lars(n_nonzero_coefs=10),
    BayesianRidge(),
    SGDClassifier(),
    RidgeClassifier(),
    LogisticRegression(),
    OrthogonalMatchingPursuit(),
    # RANSACRegressor(),
]
correction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

models = zip(names, classifiers, correction)

for name, clf, correct in models:
    regr = clf.fit(X, Y)
    # print( name,'% errors', abs(regr.predict(X)+correct-Y).sum()/(Y.sum())*100)
    print(name, '%error', procenterror(regr.predict(X), Y), 'rmsle', rmsle(regr.predict(X), Y))
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, \
        recall_score

    # Confusion Matrix
    print(name, 'Confusion Matrix')
    conf = confusion_matrix(Y, np.round(regr.predict(X)))
    label = np.sort(Y.unique())
    sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label, cmap="YlGnBu")
    plt.show()

    print('--' * 40)

    # Classification Report
    print(name, 'Classification Report')
    classif = classification_report(Y, np.round(regr.predict(X)))
    print(classif)

    # Accuracy
    print('--' * 40)
    logreg_accuracy = round(accuracy_score(Y, np.round(regr.predict(X))) * 100, 2)
    print('Accuracy', logreg_accuracy, '%')

    if name == 'DecisionTree':
        label = train.columns
        label = label[:-1].values
        important = pd.DataFrame(clf.feature_importances_, index=label, columns=['imp'])
        important.sort_values(by='imp').plot(kind='bar')
        plt.show()


    """if name == 'DecisionTree':
        from sklearn.externals.six import StringIO
        from IPython.display import Image
        from sklearn.tree import export_graphviz
        import pydotplus

        dot_data = StringIO()
        export_graphviz(regr, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        graph.write_pdf("activity.pdf")"""
