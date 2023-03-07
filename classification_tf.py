import pandas as pd
import numpy as np
import os 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split, cross_validate
from sklearn.metrics import make_scorer, recall_score, roc_curve, auc, roc_auc_score
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras import metrics, activations
from tensorflow.keras.utils import to_categorical


# Flags selection
random_labels = False
select_subset = True
tsne_generator = True
leave_30_out = True

np.random.seed(1)
list_datasets = os.listdir("DATASET_4_csv")

patient_test_list = ["112314","112854","112405","112458","112467","112675","112684","112797","112854","112863","112962","112998","113199"]

data_final = pd.DataFrame()

for i in range(len(patient_test_list)):
    patient_test = patient_test_list[i]
    patient_list = []

    for i in list_datasets:
        if patient_test in i:
            patient_list.append(i)

    print(patient_test)

    #gated = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD45_csv.csv", sep=";", decimal=",")

    ungated = pd.read_csv("DATASET_4_csv/"+patient_test+"_csv.csv", sep=";", decimal=",")

    #g = gated.copy()

    CD45 = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD45_csv.csv", sep=";", decimal=",")
    CD3 = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD3_csv.csv", sep=";", decimal=",")
    CD19 = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD19_csv.csv", sep=";", decimal=",")

    linfT = True
    linfB = False

    if linfT:
        experiment_name = "Lynfocytes_T"
        # Considero le righe di CD45 che sono compaiono anche in CD3
        df = CD45.drop_duplicates().merge(CD3.drop_duplicates(), on=CD3.columns.to_list(), how='left', indicator=True)
        df_CD3_from_CD45=df.loc[df._merge=='both',df.columns!='_merge']

        # Considero le righe che compaiono sia in CD45 che in CD3 ma che non compaiono in CD19
        df = df_CD3_from_CD45.drop_duplicates().merge(CD19.drop_duplicates(), on=CD19.columns.to_list(), how='left', indicator=True)
        df_CD19_CD3_from_CD45 = df.loc[df._merge=='left_only',df.columns!='_merge']

        g = df_CD19_CD3_from_CD45.copy()
    elif linfB:
        experiment_name = "Lymphocytes_B"
        # Considero le righe di CD45 che non compaiono anche in CD3
        df = CD45.drop_duplicates().merge(CD3.drop_duplicates(), on=CD3.columns.to_list(), how='left', indicator=True)
        df_CD3_from_CD45=df.loc[df._merge=='left_only',df.columns!='_merge']

        # Considero le righe di CD45 che non compaiono anche in CD3 ma che compaiono in CD19
        df = df_CD3_from_CD45.drop_duplicates().merge(CD19.drop_duplicates(), on=CD19.columns.to_list(), how='left', indicator=True)
        df_CD19_CD3_from_CD45 = df.loc[df._merge=='both',df.columns!='_merge']

        g = df_CD19_CD3_from_CD45.copy()

    ug = ungated.copy()

    df = ug.drop_duplicates().merge(g.drop_duplicates(), on=g.columns.to_list(), how='left', indicator=True)

    processed_df=df.loc[df._merge=='left_only',df.columns!='_merge']
    processed_df.reset_index(drop=True)
    processed_df = processed_df.sample(frac=1)

    list_of_ones = np.ones((len(g),1))
    list_of_zeros = np.zeros((len(processed_df),1))

    processed_df['label'] = list_of_zeros
    g['label'] = list_of_ones

    data = pd.concat([processed_df,g], axis=0, ignore_index=True)
    data = data.sample(frac=1)
    data_final = pd.concat([data_final,data], axis=0, ignore_index=True)
    data_final = data_final.dropna(axis=1)
    data_final = data_final.sample(frac=1)
print(data_final)

plt.figure()
sns.scatterplot(data=data_final, x="FSC-A", y="SSC-A", hue="label")
plt.savefig("data_exploration_"+experiment_name+".png", dpi=600)
plt.close()

if random_labels:
    random_labels = np.random.randint(2,size=len(data_final))
    data_final['label'][:] = random_labels

print("label 1:"+str(len(data_final[data_final['label']==1])))
X = data_final.loc[ : , data_final.columns != 'label']
y = data_final['label']
X = X.drop(["Time"], axis=1)

# Seleziono un subset per le prove esplorative in modo da avere tempi di esecuzione ridotti
if select_subset:
    X = X.iloc[0:500]
    y = y.iloc[0:500]
    print(X)
    print("label 1:"+str(len(y[y==1])))

# Grafico TSNE per visualizzare i dati 
if tsne_generator:
    tsne = TSNE(n_components=2, verbose=0, random_state=123)
    z = tsne.fit_transform(X) 
    df = pd.DataFrame()
    df["y"] = y
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    plt.figure()
    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),data=df)
    plt.title("TSNE for " + experiment_name + " experiment")
    plt.savefig("TSNE_"+experiment_name+".png", dpi=600)
    plt.close()

# Lascio fuori un 30% di dati per test successivo su modello finale
if leave_30_out:
    X, X_test_final, y, y_test_final = train_test_split(X, y, test_size=0.33, random_state=1)

# Setto il numero di fold per la crossvalidazione annidata e il numero di iterazioni per tenere sotto controllo l'overfitting
num_splits = 5
NUM_TRIALS = 5
nested_scores = np.zeros(NUM_TRIALS)


print(experiment_name)
# Eseguo il loop per ogni iterazione 
tprs = []
interp_fpr = np.linspace(0,1,1000)


# Definizione del modello di rete neurale
model = Sequential()
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(2, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', metrics.AUC(name='auc', curve='ROC', num_thresholds=1000)])  

for i in range(NUM_TRIALS):
    print("ITERATION:" + str(i))
    np.random.seed(i)
    tf.random.set_seed(i)
    cv = KFold(n_splits=num_splits, random_state=i, shuffle=True)
    fold = 0
    for train_index, test_index in cv.split(X,y):

        print("Fold:" + str(fold))

        train_ds = X.iloc[train_index]
        test_ds = X.iloc[test_index]
        train_labels = y.iloc[train_index]
        test_labels = y.iloc[test_index]  
        train_labels = to_categorical(train_labels, num_classes=2)
        test_labels = to_categorical(test_labels, num_classes=2)

        model.fit(train_ds, train_labels, epochs=100, batch_size=10, validation_split=0, shuffle=False, verbose=0)

        predict_labels = model.predict(test_ds, verbose=0)

        fpr_keras, tpr_keras, thresholds_keras_p_fold = roc_curve(test_labels[:,1], predict_labels[:,1])
        interp_tpr_keras = np.interp(interp_fpr, fpr_keras, tpr_keras)
        interp_tpr_keras[0] = 0
        interp_tpr_keras[-1] = 1
        auc_keras = auc(interp_fpr, interp_tpr_keras)
        print('ROC AUC=%0.3f' % auc_keras)
        tprs.append(interp_tpr_keras)

        fold += 1

plt.figure()
mean_tpr = np.mean(tprs,axis=0)
mean_auc = auc(interp_fpr, mean_tpr)
plt.plot(interp_fpr,mean_tpr,'b')
dummy_classifier = [0,1]
plt.plot(dummy_classifier,dummy_classifier,'r--')
plt.title('Mean AUC=%0.3f' % mean_auc)
plt.legend(['Mean ROC', 'Random Classifier'])
plt.savefig("ROCcurve_"+experiment_name+"_keras.png", dpi=600)
plt.close()

# Train final classifier final classifier
y = to_categorical(y, num_classes=2)
y_test_final = to_categorical(y_test_final, num_classes=2)
model.fit(X,y,epochs=100, batch_size=10, validation_split=0, shuffle=False, verbose=0)
predict_labels = model.predict(X_test_final, verbose=0)
fpr_keras, tpr_keras, thresholds_keras_p_fold = roc_curve(y_test_final[:,1], predict_labels[:,1])
interp_tpr_keras = np.interp(interp_fpr, fpr_keras, tpr_keras)
interp_tpr_keras[0] = 0
interp_tpr_keras[-1] = 1
plt.figure()
plt.plot(interp_fpr,interp_tpr_keras,'b')
mean_auc = auc(interp_fpr, interp_tpr_keras)
dummy_classifier = [0,1]
plt.plot(dummy_classifier,dummy_classifier,'r--')
plt.title('Mean AUC=%0.3f' % mean_auc)
plt.legend(['Mean ROC', 'Random Classifier'])
plt.savefig("ROCcurve_"+experiment_name+"_keras_Final_Classifier.png", dpi=600)
plt.close()





