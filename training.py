import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import pickle as pkl
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv("data/telecom_churn.csv")
print(data)
print(data.describe().T)

# -- SAMPLE for performance --
# data = data.sample(500)
# ----

selected_features = ['ContractRenewal', 'CustServCalls', 'DayMins', 'DataUsage']
X = data.copy()[selected_features]

y = data.copy()['Churn']

X, y = SMOTE().fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc_x = RobustScaler()
X_train_scaled = sc_x.fit_transform(X_train, y_train)
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test = pd.DataFrame(sc_x.transform(X_test), columns=X_test.columns)

clfs = [
    SVC(kernel='linear'),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    LogisticRegression()
]

for clf in clfs:
    print(f"** {clf.__class__.__name__} **")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred), "\n")

# X_train_RFE = rfe.transform(X_train)
# X_test_RFE = rfe.transform(X_test)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print(classification_report(y_test, y_pred), "\n")

# best is RF wtih 8 features (f1: 0.95, p:0.97, r: 0.98, a: 0.95)
# clf = RandomForestClassifier()
# for i in range(1, len(X_train.columns) + 1):
#     print(f"** {clf.__class__.__name__} n={i}**")
#     rfe = RFE(clf, n_features_to_select=i)
#     rfe.fit(X_train, y_train)
#     print(rfe.get_feature_names_out())
#
#     X_train_RFE = rfe.transform(X_train)
#     X_test_RFE = rfe.transform(X_test)
#     clf.fit(X_train_RFE, y_train)
#     y_pred = clf.predict(X_test_RFE)
#     print(classification_report(y_test, y_pred), "\n")

# best is RF wtih 8 features (f1: 0.95, p:0.97, r: 0.98, a: 0.95)
# clf = RandomForestClassifier()
# print(f"** {clf.__class__.__name__} n=8 **")
# rfe = RFECV(clf)
# rfe.fit(X_train, y_train)
# print(rfe.get_feature_names_out())
#
# X_train_RFE = rfe.transform(X_train)
# X_test_RFE = rfe.transform(X_test)
# clf.fit(X_train_RFE, y_train)
# y_pred = clf.predict(X_test_RFE)
# print(classification_report(y_test, y_pred), "\n")

# clf = RandomForestClassifier()
# kf = KFold(5)
# for i, (train_index, test_index) in enumerate(kf.split(X)):
#     print(f"** {clf.__class__.__name__} k={i} **")
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     print(classification_report(y_test, y_pred), "\n")

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f"** Chosen Classifier was {clf.__class__.__name__} **")
print(classification_report(y_test, y_pred), "\n")

with open("binaries/model", "wb") as model_bin:
    pkl.dump(clf, model_bin)

with open("binaries/sc_x", "wb") as sc_bin:
    pkl.dump(sc_x, sc_bin)
