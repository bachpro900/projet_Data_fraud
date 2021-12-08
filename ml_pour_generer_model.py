import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
import collections, numpy # collections.Counter(X) sert à compter les occurrences dans X de type numpy array
import datetime as dt
from imblearn.over_sampling import RandomOverSampler,SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score, recall_score, f1_score



df= pd.read_csv("https://assets-datascientest.s3-eu-west-1.amazonaws.com/de/total/fraud.csv", index_col="user_id")
df_copie = df.copy()
df.head()


#afficher les dimensions du df
df.shape

#checker si il y'a des cellules vide
print(df.isnull().sum())
print("\nil n'y a pas de valeurs manquantes")


#vérifier si il n'y a pas des id doublons
print(any(df.index.duplicated()))
print(len(df.index.unique())==df.shape[0])

#transformer les deux premières colonnes en des variables temporelles
df['signup_time'] = pd.to_datetime(df['signup_time'])
df['purchase_time'] = pd.to_datetime(df['purchase_time'])

#ajouter une nouvelle colonne qui représente la durée entre la connexion et l'achat
df['lead_time'] = pd.to_datetime(df['purchase_time']) - pd.to_datetime(df['signup_time'])

#convertir lead_time en seconds
df['lead_time']=df['lead_time'].astype('timedelta64[s]')



#replacer F par 1 et M par 0
df["sex"].replace({'F': 1, 'M': 0}, inplace=True)
#df['browser'].replace({'Chrome':0, 'Opera':1, 'Safari':2, 'IE':3 , 'FireFox':4})
df.head()

#spliter les variables catégorièlles 
dummy_variables = pd.get_dummies(data=df[['source', 'browser']], drop_first=False)



#construire un df qu'avec les variables pertinentes
df_redifined =pd.concat([df[['purchase_value', 'sex', 'age', 'lead_time', "is_fraud"]], dummy_variables], axis=1)
df_redifined.head()

def normalisation(DF):
  return (DF - DF.min()) / ( DF.max() - DF.min())
  
df_normalized = normalisation(df_redifined)
df_normalized.head()


#séprarer les données expliquatives et target

X = df_redifined.drop("is_fraud", axis=1)
y = df_redifined["is_fraud"]

#séparer les données train et test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30, random_state=42,stratify=y)

#entrainer le modeèle
model_log=KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train) #"#####################################################################################"#

#calculer la target prédite
y_pred=model_log.predict(X_test)

print("Occurence de 0 et 1 dans y_train:",collections.Counter(y_train))
print("Occurence de 0 et 1 dans y_test:",collections.Counter(y_test))
print("Occurence de 0 et 1 prédit par le modèle (y_pred):",collections.Counter(y_pred))

print(X_test.shape[0]+X_train.shape[0])
print(y_test.shape[0]+y_train.shape[0])


#calculer l'accuracy
print("l'accuracy est de: ",accuracy_score(y_test, y_pred))

#calculer la balanced accuracy
print("\nComme la Balanced accuracy est de:",balanced_accuracy_score(y_test,y_pred),"alors il y a effectivment déséquilibre des classes.")
print("L'accuracy n'a pas suffit à se prononcer sur la performance du modèle malgré qu'elle est elevée")


#matrice de confusion
print("\n Matrice de confusion:", confusion_matrix(y_test,y_pred))

tn, fp, fn, tp = confusion_matrix(y_test,y_pred).ravel()
print("\n Vrais négatifs:",tn,"\n Faux positifs:",fp,"\n Faux négatifs:",fn,"\n Vrais positifs:",tp)


#rapport de classification et focus sur le f1_score

print(classification_report(y_test, y_pred))

##############################################################
######Rééquilibrage des classe avec RandomOverSampler#########
##############################################################
oversample = RandomOverSampler(sampling_strategy='minority')
X_train_balanced, y_train_balanced = oversample.fit_resample(X_train, y_train)
print('occurance 0 et 1 dans y_train_balanced:',collections.Counter(y_train_balanced))
print("\ncompraratif entre y_train(avant oversampling) VS y_train_balanced (après oversampling):\n",collections.Counter(y_train), collections.Counter(y_train_balanced))



#entrainter le modèle avec les classes rééquilibrées

#entrainer le modeèle après rééquilbrage des classes
model_log_balanced=KNeighborsClassifier(n_neighbors=5).fit(X_train_balanced,y_train_balanced) ################################################################################""

#calculer la target prédite après rééquilbrage des classes
y_pred_balanced=model_log_balanced.predict(X_test)

print(collections.Counter(y_test))
print("\noccurences des 0 et 1 du y_predit_balanced après rééquilibrage :",collections.Counter(y_pred_balanced))


#calculer l'accuracy
print("l'accuracy est de: ",accuracy_score(y_test, y_pred_balanced))

#calculer la balanced accuracy
print("\nla Balanced accuracy est de:",balanced_accuracy_score(y_test,y_pred_balanced))


#matrice de confusion
print("\n Matrice de confusion:", confusion_matrix(y_test,y_pred_balanced))

tn, fp, fn, tp = confusion_matrix(y_test,y_pred_balanced).ravel()
print("\n Vrais négatifs:",tn,"\n Faux positifs:",fp,"\n Faux négatifs:",fn,"\n Vrais positifs:",tp)


#rapport de classification et focus sur le f1_score

print(classification_report(y_test, y_pred_balanced))




#######################################################
######Rééquilibrage des classes avec SMOTE#############
#######################################################
#rééquilibrage des classes avec SMOTE
sm = SMOTE(sampling_strategy='minority')
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)
print('occurence 0 et 1 dans le y_train_smote après oversampling:',collections.Counter(y_train_smote))
print("\ncompraratif entre y_train(avant oversampling) VS y_train_smote (après oversampling):\n",collections.Counter(y_train), collections.Counter(y_train_smote))

#entrainter le modèle avec les classes rééquilibrées

#entrainer le modeèle après rééquilbrage des classes
model_log_smote=KNeighborsClassifier(n_neighbors=5).fit(X_train_smote,y_train_smote)

#calculer la target prédite après rééquilbrage des classes
y_pred_smote=model_log_smote.predict(X_test)

print(collections.Counter(y_test))
print("\noccurences des 0 et 1 du y prédit après rééquilibrage :",collections.Counter(y_pred_smote))

#calculer l'accuracy avec smote
print("l'accuracy est de: ",accuracy_score(y_test, y_pred_smote))

#calculer la balanced accuracy
print("\nla Balanced accuracy est de:",balanced_accuracy_score(y_test,y_pred_smote))


#matrice de confusion
print("\n Matrice de confusion:", confusion_matrix(y_test,y_pred_smote))

tn, fp, fn, tp = confusion_matrix(y_test,y_pred_smote).ravel()
print("\n Vrais négatifs:",tn,"\n Faux positifs:",fp,"\n Faux négatifs:",fn,"\n Vrais positifs:",tp)


#rapport de classification et focus sur le f1_score

print(classification_report(y_test, y_pred_smote))


####################################################################
##########Rééquilibrage des classes sur l'ensemble du dataframe#####
####################################################################
oversampled_model = RandomOverSampler(sampling_strategy='minority')
XX_balanced,yy_balanced = oversampled_model.fit_resample(X, y)
XX_train_balanced, XX_test_balanced, yy_train_balanced, yy_test_balanced = train_test_split(XX_balanced, yy_balanced, test_size=0.30, random_state=42, stratify=yy_balanced)


print(collections.Counter(y_train), collections.Counter(yy_train_balanced))

print((yy_test_balanced.shape[0]+yy_train_balanced.shape[0])- df.shape[0], "lignes ont été rajoutées après l'oversampling. Le nombre initial de ligne est de 151112 !!")

#entrainter le modèle avec les classes rééquilibrées

#entrainer le modeèle après rééquilbrage des classes
model=KNeighborsClassifier(n_neighbors=5).fit(XX_train_balanced,yy_train_balanced)

#calculer la target prédite après rééquilbrage des classes
yy_pred_balanced=model.predict(XX_test_balanced)

print(collections.Counter(y_test))
print("\noccurences des 0 et 1 du y prédit après rééquilibrage :",collections.Counter(yy_pred_balanced))

#calculer l'accuracy
print("l'accuracy est de: ",accuracy_score(yy_test_balanced, yy_pred_balanced))

#calculer la balanced accuracy
print("\nla Balanced accuracy est de:",balanced_accuracy_score(yy_test_balanced,yy_pred_balanced))


#matrice de confusion
print("\n Matrice de confusion:", confusion_matrix(yy_test_balanced,yy_pred_balanced))

tn, fp, fn, tp = confusion_matrix(yy_test_balanced,yy_pred_balanced).ravel()
print("\n Vrais négatifs:",tn,"\n Faux positifs:",fp,"\n Faux négatifs:",fn,"\n Vrais positifs:",tp)


#rapport de classification et focus sur le f1_score

print(classification_report(yy_test_balanced, yy_pred_balanced))

#construire un modèle balanced avec pickle
import pickle
pickle_out = open("fraud_model.pkl", "wb")
pickle.dump (model, pickle_out)
pickle_out.close()





##############################################################
###### entrainement du modèle CatBoosterClassifier ###########
##############################################################


model_CatBooster = CatBoostClassifier(iterations=200, learning_rate=0.3,random_seed=42,verbose=False)
 
y_predicted = model_CatBooster.fit(X_train, y_train).predict(X_test)
   

Train_Accuracy = round(model_CatBooster.score(X_train, y_train), 2)
Test_Accuracy = round(model_CatBooster.score(X_test, y_test), 2)
Precision= round(precision_score(y_test, y_predicted),2)
Recall= round(recall_score(y_test, y_predicted),2)
F1_score = round(f1_score(y_test, y_predicted),2)
print(Train_Accuracy, Test_Accuracy, Precision, Recall, F1_score)



#construire un modèle catBossterClassifier avec pickle
import pickle
pickle_out = open("fraud_CatBoostClassifier_model.pkl", "wb")
pickle.dump (model_CatBooster, pickle_out)
pickle_out.close()
