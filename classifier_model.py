from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from dataset import DataSet
from sklearn.feature_selection import chi2, SelectKBest
from ShowChart import showChart
import numpy as np

data_set = DataSet()
data, label = data_set.get_train_data_set()

languages = np.array(['bg', 'mk', 'bs', 'hr', 'sr', 'cz', 'sk', 'es_AR', 'es_ES'
                         , 'pt-BR', 'pt-PT', 'id', 'my', 'xx'])

label = np.array(label)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data)

truncatedSVD = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
truncatedSVD = truncatedSVD.fit(X_train_counts)

X_r = truncatedSVD.transform(X_train_counts)
colors = ['navy', 'turquoise', 'darkorange', 'navajowhite', 'salmon'
    , 'azure', 'blue', 'brown', 'cadetblue', 'limegreen', 'maroon'
    , 'cornsilk', 'peachpuff', 'red']

showChart(X_r, label, "TruncatedSVD Language parameters ")

X_new = SelectKBest(chi2, k=2).fit_transform(X_train_counts, label)

showChart(x=X_new, y=label, title="TruncatedSVD Language parameters ")
