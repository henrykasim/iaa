import h2o
import math
import re
from h2o.automl import H2OAutoML
import pandas as pd
import numpy as np
from sklearn import preprocessing as prep
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from collections import Counter, OrderedDict

# Check column type: Categorical or Numerical/Continuous
def check_column_type(data_df):
    dt_dic = OrderedDict()
    for col in data_df.columns:
        if data_df[col].dtypes == np.float64:
            dt_dic[col] = 'numerical'
        elif data_df[col].dtypes == np.object or data_df[col].dtypes == np.bool:
            dt_dic[col] = 'categorical'
        elif ((1.*data_df[col].nunique()/data_df[col].count()) > 0.20).all():
            dt_dic[col] = 'numerical'
        else:
            dt_dic[col] = 'categorical'
    return dt_dic

# Removed unnecessary features: invariant and duplicate and contains empty values
def clean_feature(df):
    data_array = df[df.columns.values[1:]].values.T
    type_list = [np.issubdtype(arr.dtype, np.number) for arr in data_array]
    invariant = [] 
    containna = []
    # Check for duplicate column - drop column
    # Get duplicate column (all has same values)
    duplicate = []
    for i,isnumber in enumerate(type_list):
        if pd.isnull(data_array[i]).any() or pd.isna(data_array[i]).any(): # np.isnan(data_array[i]).any():
            containna.append(i)
        else:
            if isnumber:
                # invariant
                if np.var(data_array[i]) == 0:
                    invariant.append(i)
            else:
                if len(np.unique(data_array[i])) == 1:
                    invariant.append(i)
    remove = duplicate + invariant + containna
    print("Remove Following Features due to duplicate, invariant, contain empty:", remove)
    data_cols = np.delete(df.columns[1:].values,remove,None).tolist()
    keep_cols = df.columns.values.tolist()[0:1]+data_cols
    return df[keep_cols]

# Remove non-ascii character for the column names
def encode_text(value):
    return value.encode('ascii', 'ignore').decode()

# Encoding column name for clearer feature name
def encode_columns(columns):
    columns = [encode_text(i) for i in columns]
    return columns

# Preprocess the data for boolean data type - as it might cause error
def model_miner_data_preprocess(data_df):
    # handle special case where label column is boolean
    # Columns mapping (for re-reference back)
    col_dict = {}
    original_cols = data_df.columns
    if data_df[data_df.columns[0]].dtypes.name == 'bool':
        special_map = {True:'TRUE', False:'FALSE'}
        data_df[data_df.columns[0]] = data_df[data_df.columns[0]].map(special_map)
    data_df.columns = encode_columns(data_df.columns)

    for i, c in enumerate(data_df.columns):
        col_dict[c] = original_cols[i]
    return data_df, col_dict

# Preprocess the data for categorical data type 
def categorical_data_preprocess(data_df, feature_type, encoder_list=None, target=True):
    if target:
        # Special handle for categorical data
        feature_names = data_df.columns[1:].values.tolist()
    else:
        feature_names = data_df.columns.tolist()
    # encode categorical features and keep tracks
    categorical_features = [k for k,v in feature_type.items() if v == 'categorical']
    categorical_features = encode_columns(categorical_features)
    print("Categorical Features: ", categorical_features)
    categorical_index = []
    for feature in categorical_features:
        i = feature_names.index(feature)
        categorical_index.append(i)
        
    return data_df, feature_names, categorical_index

# Run Train Model
def run_modelMiner_training(data_df, feature_type, number_of_model=1, max_runtime=600):
    # metrics = ['auc', 'logloss', 'mean_per_class_error', 'rmse', 'mse']
    data_df, col_dict = model_miner_data_preprocess(data_df)
    target_df = data_df[data_df.columns[0]]
    
    # Calculate minimum number of nfolds split
    nfolds = int(np.min([x[1] for x in Counter(data_df[data_df.columns[0]].values).items()]))
    if nfolds > 10:
        nfolds = 10
    if nfolds < 2:
        nfolds = 2
    
    labels = data_df[data_df.columns[0]].values
    data_df, feature_names, categorical_index = categorical_data_preprocess(data_df, feature_type, target=True)
    
    # Initialise h2o
    # h2o.init()
    h2o.init(ip="localhost", port=54323, min_mem_size_GB=1)
    # Convert python_dataframe to h2o_dataframe
    train_h2o_df = h2o.H2OFrame(data_df[feature_names].values)
    train_h2o_df.set_names(feature_names)
    train_h2o_df['class'] = h2o.H2OFrame(target_df.tolist())
    train_h2o_df['class'] = train_h2o_df['class'].asfactor()

    # let h2o know which is categorical
    for i in categorical_index:
        train_h2o_df[i] = train_h2o_df[i].asfactor()
    # print(len(train_h2o_df.columns), train_h2o_df.columns)

    max_models = 50000
    # training with h2o.automl
    # Limit the maximum number of the model
    if number_of_model > 10:
        number_of_model = 10
    # Limit the min number of maximum runtime in sec
    #if max_runtime < 120:
    #    max_runtime < 120
    print("Run Predictive Model - #models:", number_of_model, "max_runtime:", max_runtime, "nFolds: ", nfolds)
    aml = H2OAutoML(max_models = max_models, #number_of_model, 
                    seed = 1,
                    balance_classes=False, 
                    nfolds=nfolds,
                    max_runtime_secs=max_runtime,
                    keep_cross_validation_predictions=True,
                    keep_cross_validation_models=True,
                    keep_cross_validation_fold_assignment=True,
                    project_name='H2O')
    aml.train(feature_names, 'class', training_frame=train_h2o_df)
    # aml.train((train_h2o_df.columns)[:-1], 'class', training_frame=train_h2o_df)

    # Get the list of models
    lb = aml.leaderboard
    # Columns: model_id, auc, logloss, mean_per_class_error, rmse, mse
    # Issue: as_data_frame on pyinstaller - somehow the output still appear as list (instead of dataframe). Fix: convert list to dataframe
    # models_list = lb.head(number_of_model).as_data_frame(use_pandas=False)
    models_list = lb.as_data_frame(use_pandas=False)
    headers = models_list.pop(0)
    models_df = pd.DataFrame(models_list, columns=headers)

    # Filter model for GBM, XRT or DRF
    model_path_arr = []
    model_param_arr = []
    model_fparam_arr = []
    model_summary_arr = []
    model_path_for_cv_arr = []
    model_name_for_cv_arr = []
    # if model is dataframe:
    models_df = models_df.head(number_of_model)
    for model_id in models_df['model_id'].values:
            # Get a specific model
            model = h2o.get_model(model_id)
            # save model
            model_path = h2o.save_model(model, path = "./static/file", force=True)
            model_path_arr.append(model_path)
            # get model parameter
            model_param_arr.append(str(model.params))
            # get model full parameter
            model_fparam_arr.append(str(model.full_parameters))
            # get model summary
            model_summary_arr.append(str(model.summary()))
            #model_summary_arr.append(str(model.summary()))

            # Cross validation results here
            cvmodel_path_arr = []
            cvmodel_name_arr = []
            cv_models = model.cross_validation_models() #h2o.as_list(pyModel.cross_validation_models())
            num_cv_models = len(cv_models)
            #print('Num of CV models: ', num_cv_models)
            for i, cv_model in enumerate(cv_models):
                cvmodel_path = h2o.save_model(cv_model, path = "./static/file", force=True)
                cvmodel_path_arr.append(cvmodel_path)
                cvmodel_name_arr.append(cvmodel_path.split('file')[-1].replace('\\','').replace('/',''))
            model_path_for_cv_arr.append(cvmodel_path_arr)
            model_name_for_cv_arr.append(cvmodel_name_arr)

    models_df['path']=model_path_arr
    models_df['param']=model_param_arr
    models_df['fparam']=model_fparam_arr
    models_df['summary']=model_summary_arr
    models_df['cv_name']=model_name_for_cv_arr
    models_df['cv_path']=model_path_for_cv_arr
            
    # Shutdown h2o
    # h2o.shutdown()

    return models_df, train_h2o_df, categorical_index

# Run prediction given the PM model
def run_model_prediction(data_h2o_df, model_path, cv_model_path=None):
    # Get Model Prediction
    pyModel = h2o.load_model(model_path)
    model_outcome = pyModel.predict(data_h2o_df)
    model_outcome_list = model_outcome.as_data_frame(use_pandas=False)
    headers = model_outcome_list.pop(0)
    prediction_df = pd.DataFrame(model_outcome_list, columns=headers)
    
    # cross validation results
    cv_prediction_arr = []
    if cv_model_path is not None:
        for cv_model in cv_model_path:
            cvModel = h2o.load_model(cv_model)
            cvmodel_outcome = cvModel.predict(data_h2o_df)
            cvmodel_outcome_list = cvmodel_outcome.as_data_frame(use_pandas=False)
            headers = cvmodel_outcome_list.pop(0)
            cvprediction_df = pd.DataFrame(cvmodel_outcome_list, columns=headers)
            cv_prediction_arr.append(cvprediction_df['predict'].values)

    # Original data to dataframe
    data_list = data_h2o_df.as_data_frame(use_pandas=False)
    headers = data_list.pop(0)
    data_df = pd.DataFrame(data_list, columns=headers)
    if 'class' in data_df.columns:
        # Calculate the MCC, F1score
        mcc = round(matthews_corrcoef(data_df['class'], prediction_df['predict']), 4)
        binarizer = MultiLabelBinarizer()
        binarizer.fit(data_df['class'].tolist())
        f1score = round(f1_score(binarizer.transform(data_df['class']), binarizer.transform(prediction_df['predict']), average='macro'), 4)
    
        return prediction_df, mcc, f1score, cv_prediction_arr
    
    else:
        return prediction_df, None, None, cv_prediction_arr

# Run Test prediction
def run_modelMiner_test(data_df, model_path, feature_type, cv_models_arr):
    data_df, col_dict = model_miner_data_preprocess(data_df)

    # Handle the categorical data
    #data_df, feature_names, categorical_index, categorical_names, encoder_list = categorical_data_preprocess(data_df, feature_type, encoder_list, target=False)
    data_df, feature_names, categorical_index = categorical_data_preprocess(data_df, feature_type, target=False)
    
    # Initialise h2o
    #h2o.init()
    h2o.init(ip="localhost", port=54323, min_mem_size_GB=1)
    # Convert python_dataframe to h2o_dataframe
    test_h2o_df = h2o.H2OFrame(data_df[feature_names].values)
    test_h2o_df.set_names(feature_names)
    # let h2o know which is categorical
    for i in categorical_index:
        test_h2o_df[i] = test_h2o_df[i].asfactor()
    
    # Run Model Prediction
    prediction_df, mcc, f1score, cv_prediction_arr = run_model_prediction(test_h2o_df, model_path, cv_models_arr)

    # Shutdown h2o
    # h2o.shutdown()

    return test_h2o_df, prediction_df, cv_prediction_arr
