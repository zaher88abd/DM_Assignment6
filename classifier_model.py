import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from dataset import DataSet
from sklearn.feature_selection import chi2, SelectKBest
import numpy as np
import random

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

fig, ax1 = plt.subplots()

c1 = ''
c2 = ''
c3 = ''
c4 = ''
c5 = ''
c6 = ''
c7 = ''
c8 = ''
c9 = ''
c10 = ''
c11 = ''
c12 = ''
c13 = ''
c14 = ''
comp1 = 0
comp2 = 1
# comp3 = 3
for i in random.sample(range(len(X_r)), 5000):
    if label[i] == 'bg':
        c1 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='bg', color=colors[0])
    if label[i] == 'mk':
        c2 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='mk', color=colors[1])
    if label[i] == 'bs':
        c3 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='bs', color=colors[2])
    if label[i] == 'hr':
        c4 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='hr', color=colors[3])
    if label[i] == 'sr':
        c5 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='sr', color=colors[4])
    if label[i] == 'cz':
        c6 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='cz', color=colors[5])
    if label[i] == 'sk':
        c7 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='sk', color=colors[6])
    if label[i] == 'es-AR':
        c8 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='es_AR', color=colors[7])
    if label[i] == 'es-ES':
        c9 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='es_ES', color=colors[8])
    if label[i] == 'pt-BR':
        c10 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='pt-BR', color=colors[9])
    if label[i] == 'pt-PT':
        c11 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='pt-PT', color=colors[10])
    if label[i] == 'id':
        c12 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='id', color=colors[11])
    if label[i] == 'my':
        c13 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='my', color=colors[12])
    if label[i] == 'xx':
        c14 = ax1.scatter(X_r[i, comp1], X_r[i, comp2], label='xx', color=colors[13])

scatters = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14]

ax1.legend(handles=scatters)
ax1.grid(True)
ax1.title("TruncatedSVD Language parameters ")

print(len(X_r[0]))
X_new = SelectKBest(chi2, k=2).fit_transform(X_train_counts, label)
print(X_new.shape)

fig, ax2 = plt.subplots()
plt.show()
