from data_cleanup import data_clean
from data_cleanup import split_data
from csv_import import descriptive

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support

import pandas as pd
import numpy as np

# This function creates dataframe with dummies replacing object variables
def dummied_data(df):
    # Separate the numeric columns
    cdata_num = df.select_dtypes(exclude=['object'])
    # Separate object columns
    cdata_cat = df.select_dtypes(include=['object'])
    # One-hot encode the object columns only
    cdata_cat_onehot = pd.get_dummies(cdata_cat)
    # Union the numeric columns with the one-hot encoded column
    dm = pd.concat([cdata_num, cdata_cat_onehot], axis=1)
    return dm

# This function creates logistic default model from 
def log_df_model (X_tr, y_tr):
    # Create and train the model using X_train, y_train
    return (LogisticRegression(solver='lbfgs').fit(X_tr, np.ravel(y_tr)))

# This function predicts default outcome based on model and threshold
def predict(model, X_tst, threshold=0.5):
    # Use trained model to construct preditions on the test set
    preds = model.predict_proba(X_tst)
    # Create a dataframe for the probabilities of default
    preds_df = pd.DataFrame(preds[:,1], columns = ['prob_default'])
    # Reassign loan status based on the threshold
    preds_df['loan_status'] = preds_df['prob_default'] \
        .apply(lambda x: 1 if x > threshold else 0)
    return(preds_df)
        

# This function returns classfication report
def class_rep (p_df, y_tst, target='loan_status'):
    target_names = ['Non-Default', 'Default']
    return (classification_report(y_tst, p_df[target], 
                                  target_names=target_names))

#This function draws ROC curve and returns AUROC value
def roc_auroc(prob_df, y_tst):   
    fallout, sensitivity, thresholds = roc_curve(y_tst, prob_df)
    plt.plot(fallout, sensitivity, color = 'red')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black')
    auc = roc_auc_score(y_tst, prob_df)
    textstr=('AUROC={:.4f}'.format(auc))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.6, 0.15, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.xlabel('1-Specificity (False Positive Rate')
    plt.show()
    return(auc)

# This function draws three graphs showing dependency of default recall, 
# non-default recall and model accuracy from threshold
def recall_accuracy_graph(p_df, y_tst):
    trsh=[]
    acc=[]
    df_recall=[]
    ndf_recall=[]
    eq_i=-1
    for i in range(0,100):
        trsh.append(i/100) 
        loan_status = p_df['prob_default'] \
           .apply(lambda x: 1 if x > i/100 else 0)
        scores=precision_recall_fscore_support(y_tst,loan_status)
        df_recall.append(scores[1][1])
        ndf_recall.append(scores[1][0])
        acc.append((scores[1][1]*scores[3][1]+scores[1][0]*scores[3][0]) \
            /(scores[3][0]+scores[3][1]))
        # check if ndf crosses df
        if eq_i==-1 and ndf_recall[i]>df_recall[i]: eq_i=i
    max_acc_i=np.argmax(acc)    
    ticks=np.linspace(0,1,11).tolist()
    plt.plot(trsh,df_recall)
    plt.plot(trsh,ndf_recall)
    plt.plot(trsh,acc)
    plt.xlabel("Probability Threshold")
    plt.xticks(ticks)
    plt.legend(["Default Recall","Non-default Recall","Model Accuracy"])
    
    x = [eq_i/100,max_acc_i/100,max_acc_i/100,max_acc_i/100]
    y = [acc[eq_i],acc[max_acc_i], df_recall[max_acc_i],ndf_recall[max_acc_i]]
    
    plt.vlines(x[0:2], 0, [y[0], max(y)], linestyle="dashed")
    plt.hlines(y, 0, x, linestyle="dashed")
    plt.scatter(x, y, zorder=2)
    
    plt.xlim(0,None)
    plt.ylim(0,None)
    
    for i_x, i_y in zip(x, y):
        plt.text(i_x+0.02, i_y+0.02, '({:.2f}, {:.2f})'.format(i_x, i_y))
    
    plt.show()
    return(
        {
            'threshold':trsh,
            'default_recall':df_recall,
            'non_default_recall':ndf_recall,
            'accuracy':acc
            
        })      
    
# --------------------------------------------------------------------------

# Create test and train data sets
X_train, X_test, y_train, y_test=split_data(dummied_data(data_clean), 
                                            'loan_status')
# Run default logistic default model on train data
ldm=log_df_model(X_train, y_train)  
      
#Run prediction based on the model
threshold=0.4

preds_df=predict(ldm,X_test,threshold) 

# Print the classification report
print('Threshold={}'.format(threshold))
print(class_rep(preds_df, y_test))

#Calculate accuracy of the model
print('Mean accuracy score is {:.4f}'.format(ldm.score(X_test,y_test)))

# Draw ROC curve
print('AUROC score is {:.4f}'. format(roc_auroc(preds_df['prob_default'],y_test)))

recall_accuracy_graph(preds_df, y_test)

