import pandas as pd
import numpy as np

#load csv data file
df = pd.read_csv("iris.csv")
df.head()

#separate columns
x = df[['A1','A2','A3','A4']]
y = df['NAME']
#y.head(10)
#x.head(10)

#Std data using in built fuction.
"""x_std = StandardScaler().fit_transform(x)
print(x_std)"""

#Mean 
ml=[]
for i in x:
    a = np.mean(x[i])
    ml.append(a)
    a=0
print(ml)

#standard deviation
n = len(x['A1'])
s1=0
sl=[]
for j in x:
    for i in range(n):
        s1 += ((1/n) * np.sum(((x[j][i] - (np.mean(x[j])))**2)))
    sl.append(np.sqrt(s1))

    s1=0

print(sl)

#cov matrix
C = (x-ml)/sl
print("\n...Std scaler Data...\n\n",C[:5])

cm = np.cov(C.T)
print("\n...Cov matrix...\n\n",cm)

#eigen value and vector
vals, vectors = np.linalg.eig(cm)
print("\n...Eigen Values...\n\n",vals)
print("\n...Eigen Vectors...\n\n",vectors)

#Final PCA
variances = []
for i in range(len(vals)):
    variances.append(vals[i] / np.sum(vals))
print("\n...sum of variances...\n\n",np.sum(variances))
print("\n...Variances...\n\n",variances)

proj_1 = np.dot(C,(vectors.T[0]))
proj_2 = np.dot(C,(vectors.T[1]))
final = pd.DataFrame(proj_1, columns=['PC1'])
final['PC2'] = proj_2
final['target'] = df['NAME']
print("\n...Final data set after PCA...\n\n",final)

#ploting of pca using matplot

import matplotlib.pyplot as plt
%matplotlib inline

fig = plt.figure()
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)



targets=['Iris-setosa','Iris-versicolor','Iris-virginica'] 

colors=['r','g','b']  
for target,color in zip(targets,colors):    
    indicesToKeep = final['target'] == target  
    ax.scatter(final.loc[indicesToKeep,'PC1'],
              final.loc[indicesToKeep,'PC2'],
             c=color,
             s=50)
ax.legend(targets)  
ax.grid()
