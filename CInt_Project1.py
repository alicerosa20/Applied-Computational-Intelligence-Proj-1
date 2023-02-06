## Computação Inteligente 2021/22
## Alice Rosa 90007
## Francisco Galante 90073

#%% Libraries

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import skfuzzy as fuzz
from skfuzzy import control as ctrl



#%% Functions

def metrics(y_pred, y_test, algth):
    precision=precision_score(y_test, y_pred)
    recall=recall_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    conf_matrix=confusion_matrix(y_test, y_pred)

    print('\nAlgorithm:', algth,'\nF1 Score:',f1,'\nPrecision:',precision,
          '\nRecall:',recall, '\nConfusion Matrix:\n', conf_matrix)

    return (f1, precision, recall, conf_matrix)


def MLP(X_test,h_param,X_train,y_train):
    clf = MLPClassifier(
        max_iter=1000, 
        hidden_layer_sizes=h_param[0], 
        activation=h_param[1], 
        solver=h_param[2], 
        learning_rate_init=h_param[3],
        random_state=0
        ).fit(X_train,y_train)
    return clf.predict(X_test),clf

def new_dataset_1(X_1,y_1,N1):

    X_new_1=[]
    y_new_1=[]
    for i in range(N1, len(X_1)):
        X_new_1.append(np.append(X_1[i,1],X_1[i-N1:i,0]))
        y_new_1.append(y_1[i])
    
    return np.array(X_new_1),np.array(y_new_1)

def new_dataset_2(X_2,y_2,N2):

    X_new_2=[]
    y_new_2=[]
    weight_requests= np.linspace(0.1, 1, num=N2)
    for i in range(N2, len(X_2)):
        X_new_2.append(np.append(X_2[i,1],np.average(X_2[i-N2:i,0],axis=0,weights=weight_requests)))
        y_new_2.append(y_2[i])
    
    return np.array(X_new_2),np.array(y_new_2)


def train_test_val_split(X,y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=0, shuffle=False)

    return X_train,X_val,y_train,y_val

def balance_data(X,y):
   X_minority = X[np.where(y == 1)]
   y_minority = y[np.where(y == 1)]

   X_duplicate=np.append(X,X_minority,axis=0)
   y_duplicate=np.append(y,y_minority,axis=0)
   
   return X_duplicate, y_duplicate


#%% MLP

## Read File
iot_dataset = pd.read_csv('ACI21-22_Proj1IoTGatewayCrashDataset.csv', sep=',', decimal='.')

X_iot = (iot_dataset.iloc[:,:-1]).to_numpy()
y_iot = (iot_dataset.iloc[:,-1]).to_numpy()

N_inst_1=3
## Expert 1 - Features
X_iot_new_1,y_iot_new_1 = new_dataset_1(X_iot,y_iot,N_inst_1)

N_inst_2=4
## Expert 2 - Features
X_iot_new_2,y_iot_new_2 = new_dataset_2(X_iot,y_iot,N_inst_2)

## Split train,val,test --> Expert 1
X_train_new_1,X_val_new_1,y_train_new_1,y_val_new_1=train_test_val_split(X_iot_new_1,y_iot_new_1)

## Split train,val,test --> Expert 2
X_train_new_2,X_val_new_2,y_train_new_2,y_val_new_2=train_test_val_split(X_iot_new_2,y_iot_new_2)

## Split train,val,test dataset unchanged
X_train,X_val,y_train,y_val=train_test_val_split(X_iot,y_iot)

## Balanced data for Expert 1
X_train_balanced_1,y_train_balanced_1 = balance_data(X_train_new_1,y_train_new_1)

## Balanced data for Expert 2
X_train_balanced_2,y_train_balanced_2 = balance_data(X_train_new_2,y_train_new_2)

## Hyperparameters chosen for the MLP
hyp_param=(200,'relu','adam',0.002)

## Train the models with the datasets with new features (Normal, Expert 1, Expert 2) 
y_pred_old,model1=MLP(X_val,hyp_param,X_train,y_train)
y_pred_new_1,model2=MLP(X_val_new_1,hyp_param,X_train_balanced_1,y_train_balanced_1)
y_pred_new_2,model3=MLP(X_val_new_2,hyp_param,X_train_balanced_2,y_train_balanced_2)

## Compare the tree methods (Normal, Expert 1, Expert 2) with validation set
metrics(y_pred_old, y_val, 'MLP Normal')
metrics(y_pred_new_1, y_val_new_1, 'MLP with Expert 1')
metrics(y_pred_new_2, y_val_new_2, 'MLP with Expert 2')


#%% Fuzzy System

## Define Antecedents/Consequents
load = ctrl.Antecedent(np.arange(0, 1, 0.01), 'Load')
average_number_of_Requests = ctrl.Antecedent(np.arange(0, 1, 0.01), 'Average Number of Requests')
crash = ctrl.Consequent(np.arange(0, 1, 0.01), 'Gateway Operation')

## Membership Functions of the Antecedents
average_number_of_Requests['Low'] = fuzz.trimf(load.universe, [0, 0, 0.65])
average_number_of_Requests['Medium'] = fuzz.trimf(load.universe, [0.40, 0.75, 0.80])
average_number_of_Requests['High'] = fuzz.trimf(load.universe, [0.65, 1, 1])

load['Low'] = fuzz.trimf(load.universe, [0, 0, 0.5])
load['Medium'] = fuzz.trimf(load.universe, [0.35, 0.5, 0.65])
load['High'] = fuzz.trimf(load.universe, [0.5, 1, 1])

load.view()
average_number_of_Requests.view()

## Membership Functions of the Consequent
crash['Normal Operation'] = fuzz.trimf(crash.universe, [0, 0, 0.7])
crash['Crash'] = fuzz.trimf(crash.universe, [0.3, 1, 1])

crash.view()

## Rulebase (2 Inputs & 3 Linguistic Terms --> 9 Rules)
rule1 = ctrl.Rule(load['Low'] & average_number_of_Requests['Low'], crash['Normal Operation'])
rule2 = ctrl.Rule(load['Low'] & average_number_of_Requests['Medium'], crash['Normal Operation'])
rule3 = ctrl.Rule(load['Low'] & average_number_of_Requests['High'], crash['Normal Operation'])

rule4 = ctrl.Rule(load['Medium'] & average_number_of_Requests['Low'], crash['Normal Operation'])
rule5 = ctrl.Rule(load['Medium'] & average_number_of_Requests['Medium'], crash['Normal Operation'])
rule6 = ctrl.Rule(load['Medium'] & average_number_of_Requests['High'], crash['Normal Operation'])

rule7 = ctrl.Rule(load['High'] & average_number_of_Requests['Low'], crash['Normal Operation'])
rule8 = ctrl.Rule(load['High'] & average_number_of_Requests['Medium'], crash['Crash'])
rule9 = ctrl.Rule(load['High'] & average_number_of_Requests['High'], crash['Crash'])

## Crash Rules Control System
crash_rules = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, 
                                rule6, rule7, rule8, rule9])
crash_result = ctrl.ControlSystemSimulation(crash_rules)

## Passing inputs to the Control System using the Antecedent labels
crash_result.input['Load'] = X_iot_new_2[:,0]
crash_result.input['Average Number of Requests'] = X_iot_new_2[:, 1]

## Final Fuzzy System Result
crash_result.compute()
result = np.round(crash_result.output['Gateway Operation'])
metrics(result,y_iot_new_2, 'Fuzzy System w/ Expert 2')


#%% Generalization - Final evaluation with new test set

iot_dataset = pd.read_csv('ACI_Proj1_TestSet.csv', sep=',', decimal='.')

X_iot = (iot_dataset.iloc[:,:-1]).to_numpy()
y_iot = (iot_dataset.iloc[:,-1]).to_numpy()

N_inst_1=3
## Expert 1 - Features
X_iot_test_1,y_iot_test_1 = new_dataset_1(X_iot,y_iot,N_inst_1)

N_inst_2=4
## Expert 2 - Features
X_iot_test_2,y_iot_test_2 = new_dataset_2(X_iot,y_iot,N_inst_2)

## Final prediction with test set for Expert 1
y_pred_final_1=model2.predict(X_iot_test_1)

metrics(y_pred_final_1,y_iot_test_1,'MLP Final Expert 1')

## Final prediction with test set for Expert 2
y_pred_final_2=model3.predict(X_iot_test_2)

metrics(y_pred_final_2,y_iot_test_2,'MLP Final Expert 2')

## Final prediction with test set for Fuzzy System
crash_result.input['Load'] = X_iot_test_2[:,0]
crash_result.input['Average Number of Requests'] = X_iot_test_2[:, 1]

crash_result.compute()

result = np.round(crash_result.output['Gateway Operation'])
metrics(result,y_iot_test_2, 'Fuzzy System Final')