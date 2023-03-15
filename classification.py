import pandas as pd
import numpy as np
import os 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, roc_curve, auc, roc_auc_score
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from xgboost import XGBClassifier

# Flags selection
random_labels = False
select_subset = True
tsne_generator = True
leave_30_out = True

# Model selection
svm_model = False
XGB_model = True
tree_model = False

np.random.seed(1)
list_datasets = os.listdir("DATASET_4_csv")

patient_test_list = ["112314","112854","112405","112458","112467","112675","112684","112797","112854","112863","112962","112998","113199"]

data_final = pd.DataFrame()

for i in range(len(patient_test_list)):
    patient_test = patient_test_list[i]
    patient_list = []

    for j in list_datasets:
        if patient_test in j:
            patient_list.append(j)

    print(patient_test)

    #gated = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD45_csv.csv", sep=";", decimal=",")

    ungated = pd.read_csv("DATASET_4_csv/"+patient_test+"_csv.csv", sep=";", decimal=",")

    #g = gated.copy()

    CD45 = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD45_csv.csv", sep=";", decimal=",")
    CD3 = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD3_csv.csv", sep=";", decimal=",")
    CD19 = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD19_csv.csv", sep=";", decimal=",")
    CD8 = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD8_csv.csv", sep=";", decimal=",")
    CD4 = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD4_csv.csv", sep=";", decimal=",")

    linfT = False
    linfB = False
    linfTtox = True

    if linfT:
        experiment_name = "Lymphocytes_T"
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
    if linfTtox:
        experiment_name = "Lymphocytes_T_cytotox"
        # Considero le righe di CD45 che sono compaiono anche in CD3
        df = CD45.drop_duplicates().merge(CD3.drop_duplicates(), on=CD3.columns.to_list(), how='left', indicator=True)
        df_CD3_from_CD45=df.loc[df._merge=='both',df.columns!='_merge']

        # Considero le righe che compaiono in CD45, in CD3 e in CD8
        df = df_CD3_from_CD45.drop_duplicates().merge(CD8.drop_duplicates(), on=CD8.columns.to_list(), how='left', indicator=True)
        df_CD8_CD3_from_CD45 = df.loc[df._merge=='both',df.columns!='_merge']

        # Considero le righe che compaiono in CD45, in CD3 e in CD8 ma non in CD4
        df = df_CD8_CD3_from_CD45.drop_duplicates().merge(CD4.drop_duplicates(), on=CD4.columns.to_list(), how='left', indicator=True)
        df_CD4_CD8_CD3_from_CD45 = df.loc[df._merge=='left_only',df.columns!='_merge']

        g = df_CD4_CD8_CD3_from_CD45.copy()

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
sns.scatterplot(data=data_final, x="SSC-A", y="CD45", hue="label")
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
    X = X.iloc[0:1000]
    y = y.iloc[0:1000]
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

##########################################################################################
##########################################################################################
##########################################################################################

# Inizializzo il modello e gli iperparamentri da esplorare all'interno della gridsearch
# SVM
if svm_model:
    p_grid = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}
    model = SVC(kernel="linear")
    model_name = "SVM"

# XGBoost
if XGB_model:
    p_grid = {  "gamma":[0, 0.1, 0.2,0.3,0.4,0.5],
                "max_depth": [3,5,10],
                "n_estimators":[5,10, 20, 100],
                "ubsample": [0.25, 0.5, 1],
                "verbosity": [0]
            }
    model = XGBClassifier(silent=True)
    model_name = "XGB"

# Decision Tree
if tree_model:
    p_grid = {  "criterion":['gini','entropy'],
                "max_depth":[2,4,6,8,10,12]

             }
    model = tree.DecisionTreeClassifier()
    model_name = "DecisionTree"

###########################################################################################
###########################################################################################
###########################################################################################

# Setto il numero di fold per la crossvalidazione annidata e il numero di iterazioni per tenere sotto controllo l'overfitting
num_splits = 5
NUM_TRIALS = 5
nested_scores = np.zeros(NUM_TRIALS)

# Setto le tipologie di score di interesse per l'esperimento, avremo poi queste informazioni in uscita dalla crossvalidazione annidata
myscoring = {'bal_acc': 'balanced_accuracy',
                        'roc_auc': 'roc_auc',
                        'ave_pre': 'average_precision',
                        'sensitivity': 'recall'
                        }

# Inizializzo i vettori contenenti i risultati in termini di score della crossvalidazione annidata per training e test 
bal_acc_train_scores = np.zeros((NUM_TRIALS,1))
roc_auc_train_scores = np.zeros((NUM_TRIALS,1))
ave_pre_train_scores = np.zeros((NUM_TRIALS,1))
bal_acc_test_scores = np.zeros((NUM_TRIALS,1))
roc_auc_test_scores = np.zeros((NUM_TRIALS,1))
ave_pre_test_scores = np.zeros((NUM_TRIALS,1))
mean_fpr = np.linspace(0, 1, 1000)
tprs = []

print(experiment_name)
# Eseguo il loop per ogni iterazione 
for i in range(NUM_TRIALS):
    print("ITERATION:"+str(i))
    np.random.seed(i)
    # Definisco il ciclo interno ed esterno della crossvalidazione annidata
    inner_cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=i)
    outer_cv = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=i)

    # ESECUZIONE DELLA CROSSVALIDAZIONE ANNIDATA
    # Il classificatore è una gridserch che crossvalida sul ciclo interno 
    clf = GridSearchCV(estimator=model, param_grid=p_grid, scoring='roc_auc', n_jobs=-1, refit=True, cv=inner_cv, verbose=0, return_train_score=True)
    # Il classificatore viene crossvalidato nel ciclo esterno
    nested_score = cross_validate(clf, X=X, y=y, cv=outer_cv, return_train_score=True, return_estimator=True, scoring=myscoring)
    # Salvo nei vettori perogni iterazione i valori medi tra i vari fold degli score ottenuti in train e test set durante la crossvalidazione e li stampo a video
    bal_acc_train_scores[i] = np.mean(nested_score['train_bal_acc'])
    roc_auc_train_scores[i] = np.mean(nested_score['train_roc_auc'])
    ave_pre_train_scores[i] = np.mean(nested_score['train_ave_pre'])
    print('Train: bal_acc ' + str( bal_acc_train_scores[i]))
    print('Train: roc_auc ' + str(roc_auc_train_scores[i]))
    print('Train: ave_pre ' + str(ave_pre_train_scores[i]))
    bal_acc_test_scores[i] = np.mean(nested_score['test_bal_acc'])
    roc_auc_test_scores[i] = np.mean(nested_score['test_roc_auc'])
    ave_pre_test_scores[i] = np.mean(nested_score['test_ave_pre'])
    print('Test: bal_acc ' + str( bal_acc_test_scores[i]))
    print('Test: roc_auc ' + str(roc_auc_test_scores[i]))
    print('Test: ave_pre ' + str(ave_pre_test_scores[i]))

    # TRUE POSITIVE RATE COMPUTATION FOR EACH OUTER LOOP (TEST SET)
    j = 0
    for train_index, test_index in inner_cv.split(X, y):
        print("Split:", j)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        classifier_fold = nested_score['estimator'][j].best_estimator_

        classifier_fold.fit(X_train, y_train)
             
        y_pred_labels = classifier_fold.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_labels)
        roc_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        j += 1

    plt.figure()
    plt.plot([0, 1], [0, 1], '--', color='r', label='Random classifier', lw=2, alpha=0.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.title('Mean AUC=%0.3f' % mean_auc)
    plt.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC', lw=2, alpha=0.8)

    ## Standard deviation computation
    std_tpr = np.std(tprs, axis=0)
    tprs_upper_std = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower_std = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower_std, tprs_upper_std, color='green', alpha=.2,label=r'$\pm$ 1 SD')

    ## 99.9% CI computation
    z = 3.291
    SE = std_tpr / np.sqrt(NUM_TRIALS * 5)
    tprs_upper_95CI = mean_tpr + (z * SE)
    tprs_lower_95CI = mean_tpr - (z * SE)
    plt.fill_between(mean_fpr, tprs_lower_95CI, tprs_upper_95CI, color='grey', alpha=.5,label=r'$\pm$ 99.9% CI')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("ROCcurve_"+experiment_name+"_"+model_name+".png", dpi=600)
    plt.close()

# FINAL MODEL
print("Training final classifier...")
# Definisco la gridsearch per il modello finale, prendo gli stessi parametri del loop interno della crossvalidazione annidata
clf_final = GridSearchCV(estimator=model, param_grid=p_grid, scoring='roc_auc', n_jobs=-1, refit=True, cv=inner_cv, verbose=0, return_train_score=True)
# Fitto il modello sui dati che abbiamo usato per la crossvalidazione annidata
clf_final.fit(X,y)
# Estraggo il modello migliore tra quelli valutati con la gridsearch precedente
best_model = clf_final.best_estimator_
print("Best final estimator:")
print(best_model)
if leave_30_out:
    # Eseguo la predizione sui dati tenuti fuori prima della crossvalidazione annidata
    y_final_pred_labels = best_model.predict(X_test_final)
    # Calcolo false positive rate e true positive rate per la roc dando come argomento le label vere e quelle predette dal modello appena trainato
    fpr, tpr, thresholds = roc_curve(y_test_final, y_final_pred_labels)
    # Calcoli per il plotting della roc
    roc_auc = auc(fpr, tpr)
    print("roc_auc final model: " + str(np.round(roc_auc,3)))
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    plt.figure()
    plt.plot([0, 1], [0, 1], '--', color='r', label='Random classifier', lw=2, alpha=0.8)
    interp_tpr[-1] = 1.0
    roc_auc = auc(mean_fpr, interp_tpr)
    plt.title('Final Classifier AUC=%0.3f' % roc_auc)
    plt.plot(mean_fpr, interp_tpr, color='b', label='Mean ROC', lw=2, alpha=0.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("ROCcurve_"+experiment_name+"_"+model_name+"_final_classificator.png", dpi=600)
    plt.close()




#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
#clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
#print(clf.score(X_test,y_test))
