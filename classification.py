import pandas as pd
import numpy as np
import os 
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, train_test_split, cross_validate
from sklearn.metrics import make_scorer, recall_score, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

np.random.seed(1)
list_datasets = os.listdir("DATASET_4_csv")

patient_test_list = ["112314"]#,"112854","112405"]

data_final = pd.DataFrame()

for i in range(len(patient_test_list)):
    patient_test = patient_test_list[i]
    patient_list = []

    for i in list_datasets:
        if patient_test in i:
            patient_list.append(i)

    print(patient_test)

    gated = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD45_csv.csv", sep=";", decimal=",")

    ungated = pd.read_csv("DATASET_4_csv/"+patient_test+"_csv.csv", sep=";", decimal=",")

    g = gated.copy()
    #ug = ungated.copy()

    #df = ug.drop_duplicates().merge(g.drop_duplicates(), on=g.columns.to_list(), how='left', indicator=True)
    #
    #
    #processed_df=df.loc[df._merge=='left_only',df.columns!='_merge']
    #processed_df.reset_index(drop=True)
    #processed_df = processed_df.sample(frac=1)
    #print("Zero len:" + str(len(processed_df)))#
    #
    #list_of_ones = np.ones((len(g),1))
    #list_of_zeros = np.zeros((len(processed_df),1))#
    #
    #processed_df['label'] = list_of_zeros
    #g['label'] = list_of_ones
    #
    #data = pd.concat([processed_df,g], axis=0, ignore_index=True)
    #data = data.sample(frac=1)
    #
    #X = data.loc[ : , data.columns != 'label']
    #y = data['label']

    CD45 = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD45_csv.csv", sep=";", decimal=",")
    CD3 = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD3_csv.csv", sep=";", decimal=",")
    CD19 = pd.read_csv("DATASET_4_csv/Gated "+patient_test+"_CD19_csv.csv", sep=";", decimal=",")

    # Considero le righe di CD45 che sono compaiono anche in CD3
    df = CD45.drop_duplicates().merge(CD3.drop_duplicates(), on=CD3.columns.to_list(), how='left', indicator=True)
    df_CD3_from_CD45=df.loc[df._merge=='both',df.columns!='_merge']

    # Considero le righe di CD3 che non compaiono in CD19
    df = df_CD3_from_CD45.drop_duplicates().merge(CD19.drop_duplicates(), on=CD19.columns.to_list(), how='left', indicator=True)
    df_CD19_CD3_from_CD45 = df.loc[df._merge=='left_only',df.columns!='_merge']

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
print(data_final)
print("label 1:"+str(len(data_final[data_final['label']==1])))
X = data_final.loc[ : , data_final.columns != 'label']
y = data_final['label']
X = X.drop(["Time"], axis=1)

X = X.iloc[0:2000]
y = y.iloc[0:2000]

p_grid = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}
svm = SVC(kernel="linear")
NUM_TRIALS = 5
nested_scores = np.zeros(NUM_TRIALS)

myscoring = {'bal_acc': 'balanced_accuracy',
                        'roc_auc': 'roc_auc',
                        'ave_pre': 'average_precision',
                        'sensitivity': 'recall'
                        }

bal_acc_train_scores = np.zeros((NUM_TRIALS,1))
roc_auc_train_scores = np.zeros((NUM_TRIALS,1))
ave_pre_train_scores = np.zeros((NUM_TRIALS,1))
bal_acc_test_scores = np.zeros((NUM_TRIALS,1))
roc_auc_test_scores = np.zeros((NUM_TRIALS,1))
ave_pre_test_scores = np.zeros((NUM_TRIALS,1))
mean_fpr = np.linspace(0, 1, 1000)
tprs = []

# Loop for each trial
for i in range(NUM_TRIALS):
    print("ITERATION:"+str(i))
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)
    clf = GridSearchCV(estimator=svm, param_grid=p_grid, scoring='roc_auc', n_jobs=-1, refit=True, cv=inner_cv, verbose=0, return_train_score=True)
    nested_score = cross_validate(clf, X=X, y=y, cv=outer_cv, return_train_score=True, return_estimator=True, scoring=myscoring)
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
    j = 0

    for train_index, test_index in inner_cv.split(X, y):
        print("Split:", j)
            
        ## TRUE POSITIVE RATE COMPUTATION FOR EACH OUTER LOOP (TEST SET)
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
    plt.title('AUC=%0.3f' % mean_auc)
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
    #plt.axis('square')
    #plt.savefig(dir_name + "/img/"+model_name+"_ROCcurve" + str(int(float(sw))) + ".png", dpi=600)
    plt.savefig("ROCcurve.png", dpi=600)
    plt.close()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
#clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
#print(clf.score(X_test,y_test))
