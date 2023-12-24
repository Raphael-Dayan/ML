### Vamos tentar aquela random forest customizada para datasets desbalanceados
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.base import clone
#### Cada arvore vê todos os positivos e uma amostra aleatoria dos negativos. #####

# ajeitando parametros
n_clfs = 100
base_clf = DecisionTreeClassifier(max_depth=4, splitter='random', random_state = 42)

# separando indices
indices_True = y_train[y_train == True].index
indices_False = y_train[y_train == False].index

ensemble = [clone(base_clf) for _ in range(n_clfs)]


# cada árvore vê todos os fundos e uma amostra de mesmo tamanho dos demais macroprodutos
for clf_index, clf in enumerate(ensemble):

    # print(tree_index) # o indice está mudando.
  
    # gerando dataset - o dataset está mudando
    y_RF = pd.concat([y_train.loc[indices_False].sample(n = len(indices_True), random_state = clf_index),
                      y_train.loc[indices_True]]) # todos os fundos
    X_RF = X_train.loc[y_RF.index]
  
    # embaralhando
    X_RF, y_RF = shuffle(X_RF,y_RF)
  
    # fitando
    clf.fit(X_RF,y_RF)

    print(f"Fitted clf {clf_index+1} of {n_clfs}")


Y_pred_test = np.empty([n_clfs, len(X_test)])
for j, classifier in enumerate(ensemble)
    Y_pred_test[j] = classifier.predict(X_test)
    
Y_pred_train = np.empty([n_clfs, len(X_train)])
for j, classifier in enumerate(ensemble)
    Y_pred_train[j] = classifier.predict(X_train)
    
Y_pred_val = np.empty([n_clfs, len(X_val)])
for j, classifier in enumerate(ensemble)
    Y_pred_val[j] = classifier.predict(X_val)

from scipy.stats import mode
print(f"Train\n{classification_report(y_true = y_train, y_pred = np.array(mode(Y_pred_train))[0].reshape(-1,1))}\n\n")
print(f"Test\n{classification_report(y_true = y_test, y_pred = np.array(mode(Y_pred_test))[0].reshape(-1,1))}\n\n")
print(f"Val\n{classification_report(y_true = y_val, y_pred = np.array(mode(Y_pred_val))[0].reshape(-1,1))}")



