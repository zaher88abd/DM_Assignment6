from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from dataset import DataSet
from ShowChart import showChart


def getChiChart(numComp=2):
    data_set = DataSet()
    data, label = data_set.get_train_data_set()

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)

    X_new = SelectKBest(chi2, k=numComp).fit_transform(X_train_counts, label)

    showChart(x=X_new, y=label, title="Chart Chi for best Component ")
