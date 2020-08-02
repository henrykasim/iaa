import json
import pandas as pd 
import numpy as np
import datetime
import json
import csv
import re
import string
from collections import OrderedDict
from time import mktime

#from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files import File
from itertools import chain

from rest_framework.authentication import TokenAuthentication
from rest_framework.response import Response
from rest_framework.renderers import JSONRenderer
from rest_framework import status, viewsets
from rest_framework.views import APIView
from rest_framework.parsers import FormParser, FileUploadParser, MultiPartParser

# Predictive Model
from core.modelMiner import encode_text, check_column_type, clean_feature
from core.modelMiner import run_modelMiner_training, run_model_prediction, run_modelMiner_test

from core.models import PredictiveModel, PredictiveModelFeature, PredictiveModelSample, PMMachineLearning, PMMLCV, PMMLPrediction
from core.serializers import PMMLPredictionSerializer, PredictiveModelFeatureSerializer, PredictiveModelSampleSerializer, PMMachineLearningSerializer, PredictiveModelSerializer, PredictiveModelDetailSerializer



### Helper Functions ###

# Normalize item list based on max-min approach
def normalize(array):
    return (array-np.min(array))/(np.max(array)-np.min(array))

# Normalize column name
def normalize_name(col):
    col = col.replace('-', '_')
    col = re.sub("[{0}\s]+".format(string.punctuation), " ", col)
    col = re.sub(" +", " ", col).strip()
    col = col.encode('ascii', 'ignore').decode()
    return col.lower().replace(' ', '_')

# Read input csv file format
def read_csv_file(file_path):
    # Check if the file is in zip file
    if file_path.endswith('.zip'):
        data_df = pd.read_csv(file_path, compression='zip', index_col=0)
    else:
        data_df = pd.read_csv(file_path, index_col=0)
    # normalize column name
    data_df.columns = [normalize_name(i) for i in data_df.columns]
    # Change the index name to string
    data_df.index = data_df.index.map(str)
    # Remove duplicate (row) indexes
    data_df = data_df[~data_df.index.duplicated(keep='first')]
    return data_df

# Save output to csv file format
def to_csv_file(data_df, file_path):
    # Check if the file is in zip file
    if file_path.endswith('.zip'):
        data_df.to_csv(file_path, compression='zip')
    else:
        data_df.to_csv(file_path)


###
# Predictive Model - Start 
###

# /pmdata/ - data uploaded (initial analysis)
class PMDataAPIView(APIView):
    renderer_classes = (JSONRenderer, )
    #parser_classes = (MultiPartParser, FormParser)
    authentication_classes = (TokenAuthentication,)

    def get(self, request, *args, **kwargs):
        queryset = PredictiveModel.objects.all()
        serializer_class = PredictiveModelSerializer(queryset, many=True, context={'request': request})
        return Response(serializer_class.data)

    def get_data(self, uid):
        data = PredictiveModel.objects.filter(uid=uid).first()
        if data:
            return data
        return None

    def post(self, request, *args, **kwargs):
        time_start = datetime.datetime.now()
        print(time_start)
        if 'uid' not in request.data:
            uid = mktime(time_start.timetuple())
            request.data['uid'] = uid
        else:
            uid = request.data['uid']
            
        # Check in case uid exist - remove if exist
        data = self.get_data(uid)
        if data is not None: 
            data.delete()

        serializer = PredictiveModelSerializer(data=request.data, context={'request': request})
        if serializer.is_valid():
            serializer.save()
            # Read the uploaded file - capture the header
            data = PredictiveModel.objects.get(pk=uid)
            try:
                # Get the data
                data_df = read_csv_file(data.file.path)
                # Clean data (for nan and invariance)
                data_df = clean_feature(data_df)
                # data_df = data_df.round(12)
                # Write back the cleaned data to the file
                to_csv_file(data_df, data.file.path)

                # Check the target column task (if the task not specified)
                if not('task' in request.data) or data.task == '':
                    data.task = 'classification'
                    target_datatype = check_column_type(data_df.iloc[:,0:1])
                    if len(target_datatype) > 0 and target_datatype[data_df.columns[0]] == 'numerical':
                        data.task = 'regression'
                    data.save()

                # Adding Features
                # Check columns data type: ignore the first column (target)
                datatype_dict = check_column_type(data_df.iloc[:,1:])
                # Bulk_create
                f_insert_arr = []
                for f, dt in datatype_dict.items():
                    feature = PredictiveModelFeature(pm=data, feature=f, type=dt)
                    f_insert_arr.append(feature)
                    # feature.save()
                PredictiveModelFeature.objects.bulk_create(f_insert_arr)

                # Adding Samples
                # Check row_index
                sample_insert_arr = []
                #for index, row in data_df.iterrows():
                for index in data_df.index:
                    stype = "train"
                    sample = PredictiveModelSample(pm=data, sample=index, target=str(data_df[data_df.columns[0]][index]), type=stype) 
                    sample_insert_arr.append(sample)
                    # sample.save()
                PredictiveModelSample.objects.bulk_create(sample_insert_arr)

            except None:
                pass

            print(datetime.datetime.now())
            return Response(serializer.data) #, status=status.HTTP_201_CREATED)
        return HttpResponse("Error") # Response(serializer.errors) #, status=status.HTTP_400_BAD_REQUEST)


# /pmdata/uid - @get: display dataset detail
# /pmdata/uid - @post: submit test dataset
class PMDataAPIViewDetail(APIView):
    renderer_classes = (JSONRenderer, )
    def get_object(self, uid):
        try:
            return PredictiveModel.objects.get(pk=uid)
        except PredictiveModel.DoesNotExist:
            return None

    def get_features(self, Data):
        return PredictiveModelFeature.objects.filter(pm=Data)

    def get_sample(self, Data, name):
        try:
            return PredictiveModelSample.objects.filter(pm=Data, sample=name).last()
        except PredictiveModelSample.DoesNotExist:
            return None

    def get_models(self, uid):
        try:
            return PMMachineLearning.objects.filter(uid=uid)
        except PMMachineLearning.DoesNotExist:
            return None

    def get_cv_models(self, pmml):
        try:
            return PMMLCV.objects.filter(ml=pmml)
        except PMMLCV.DoesNotExist:
            return None

    def get(self, request, uid, format=None):
        data = self.get_object(uid)
        if data is not None:
            serializer = PredictiveModelSerializer(data)
            return Response(serializer.data)
        else:
            return Response([])

    def delete(self, request, uid, format=None):
        data = self.get_object(uid)
        if data is not None:
            data.delete()
        try:
            SelectedParameter.objects.filter(uid=uid, type='PM').delete()
            PMMachineLearning.objects.filter(uid=uid).delete()
        except PMMachineLearning.DoesNotExist:
            pass
        queryset = PredictiveModel.objects.all()
        serializer_class = PredictiveModelSerializer(queryset, many=True, context={'request': request})
        return Response(serializer_class.data)
    
    def put(self, request, uid, format=None):
        data = self.get_object(uid)
        if data is not None:
            # Get the request data
            if 'task' in request.data:
                data.task = request.data['task']
            if 'nModel' in request.data:
                data.nModel = request.data['nModel']
            if 'maxRuntime' in request.data:
                data.maxRuntime = request.data['maxRuntime']
            if 'modelName' in request.data:
                data.model = request.data['modelName']
            if 'runAdvanced' in request.data:
                data.runAdvanced = request.data['runAdvanced']
            data.save()

            serializer = PredictiveModelSerializer(data)
            return Response(serializer.data)
        return Response([])

    # When user upload test data
    def post(self, request, uid, *args, **kwargs):
        print(datetime.datetime.now())
        data = self.get_object(uid)
        if data is None:
            return HttpResponse("Error: UID does not exist")

        # Get the previous (Training) data
        data_df = read_csv_file(data.file.path)
        # Get the new data - Read file and construct list (Preprocessing)
        headers = []
        new_data_arr = []
        for i, line in enumerate(request.data['file']):
            if i == 0:
                headers = line.decode("utf8").rstrip().split(",")
            else:
                new_data_arr.append(line.decode("utf8").replace("\"","").replace("`","").replace("'","").rstrip().split(","))
        # Construct dataframe
        test_data_df = pd.DataFrame(new_data_arr, columns=headers) # read_csv_file(csv.DictReader(request.data['file'].read()))
        test_data_df.columns = [normalize_name(i) for i in test_data_df.columns]
        # Set first column as index
        test_data_df = test_data_df.set_index(test_data_df.columns[0])
        
        # Drop feature that not exist in training-set
        feature_names = []
        categorical_features = []
        features = self.get_features(data)
        for f in features:
            feature_names.append(f.feature)
            if f.type == 'categorical':
                categorical_features.append(f.feature)
        test_data_df = test_data_df[feature_names]

        # Save the new sample as test set
        # Adding Samples without existing index in previous data
        sample_insert_arr = []
        new_sample_index_arr = []
        for index, row in test_data_df.iterrows():
            if index not in data_df.index.values:
                stype = "test"
                sample = PredictiveModelSample(pm=data, sample=index, type=stype) #, target=str(row[0]))
                sample_insert_arr.append(sample)
                new_sample_index_arr.append(index)
        if len(sample_insert_arr)>0:
            PredictiveModelSample.objects.bulk_create(sample_insert_arr)
        #print(len(data_df.index.values), len(data_df.columns), len(test_data_df.index.values), len(test_data_df.columns))
        

        # Concatenate past data with test data
        merge_df = pd.concat([data_df, test_data_df])
        #print(merge_df.index.values, merge_df.columns, len(merge_df.index.values), len(merge_df.columns))
        
        # Write the merged data to the file
        to_csv_file(merge_df, data.file.path)
        
        # Get the column type
        col_type = OrderedDict()
        features = self.get_features(data)
        for f in features:
            col_type[f.feature] = f.type

        # Load Each PMML Model
        models = self.get_models(uid)
        if models is not None:
            for m in models:
                model_path = m.modelfile.path
                # Get the cv_model
                cv_models = self.get_cv_models(m)
                cv_models_arr = []
                for cv in cv_models:
                    cv_models_arr.append(cv.modelfile.path)

                # Make Prediction (on test_data_df only)
                test_h2o_df, prediction_df, cv_prediction_arr = run_modelMiner_test(test_data_df, model_path, col_type, cv_models_arr)
                
                # Store Prediction
                PMMLPrediction_arr = []
                numberCV = len(cv_prediction_arr)
                # Save the model prediction for each sample
                for i,sname in enumerate(test_data_df.index.values):
                    sample = self.get_sample(data, sname)
                    if sample is not None and len(prediction_df['predict']) >= i:
                        pred_value = str(prediction_df['predict'][i])
                        # Get confidence from cv prediction
                        confidence_count = len([cv[i] for cv in cv_prediction_arr if str(cv[i]) == pred_value])
                        confidence = round((1.0*confidence_count)/(1.0*numberCV)*100.0,2)
                        pmml_pred = PMMLPrediction(ml=m, sample=sample, value=pred_value, confidence=confidence)
                        PMMLPrediction_arr.append(pmml_pred)
                if len(PMMLPrediction_arr)>0:
                    PMMLPrediction.objects.bulk_create(PMMLPrediction_arr)

        else:
            print("Model not found!")

        print(datetime.datetime.now())
        serializer = PredictiveModelSerializer(data)
        return Response(serializer.data)


# -- - Display dataset features
class PMFeatureViewSet(viewsets.ModelViewSet):
    renderer_classes = (JSONRenderer, )
    queryset = PredictiveModelFeature.objects.all()
    serializer_class = PredictiveModelFeatureSerializer

# -- - Display dataset samples
class PMSampleViewSet(APIView):
    renderer_classes = (JSONRenderer, )
    def get_sample(self, sid):
        try:
            return PredictiveModelSample.objects.get(pk=sid)
        except PredictiveModelSample.DoesNotExist:
            return None

    def put(self, request, sid, format=None):
        sample = self.get_sample(sid)
        if sample is not None:
            if 'target' in request.data:
                sample.target = request.data['target']
                sample.save()

            serializer = PredictiveModelSampleSerializer(sample)
            return Response(serializer.data)
        return Response([])

# /runPM/uid - Run predictive model for given uid
class RunPredictiveModelAPI(APIView):
    renderer_classes = (JSONRenderer, )
    def get_object(self, uid):
        try:
            return PredictiveModel.objects.get(pk=uid)
        except PredictiveModel.DoesNotExist:
            return None

    def get_features(self, Data):
        return PredictiveModelFeature.objects.filter(pm=Data)

    def get_samples(self, Data):
        return PredictiveModelSample.objects.filter(pm=Data)

    def get_sample(self, Data, name):
        try:
            return PredictiveModelSample.objects.filter(pm=Data, sample=name).last()
        except PredictiveModelSample.DoesNotExist:
            return None

    def get_feature(self, Data, name):
        try:
            return PredictiveModelFeature.objects.filter(pm=Data, feature=name).last()
        except PredictiveModelFeature.DoesNotExist:
            return None
        

    def get(self, request, uid, format=None):
        print('Start:', datetime.datetime.now())
        before = datetime.datetime.now()
        time_now = datetime.datetime.now()
        data = self.get_object(uid)
        if data is None:
            return HttpResponse("Error: UID does not exist")
        try:
            PMMachineLearning.objects.filter(uid=uid).delete()
        except PMMachineLearning.DoesNotExist:
            pass

        if request.GET and request.GET.get('nModel'):
            data.nModel = int(request.GET.get('nModel'))
            print("Number of Model specified: "+str(data.nModel))
        if request.GET and request.GET.get('maxRuntime'):
            data.maxRuntime = int(request.GET.get('maxRuntime'))
            print("Max Runtime specified: "+str(data.maxRuntime))
        if request.GET and request.GET.get('runAdvanced'):
            data.runAdvanced = int(request.GET.get('runAdvanced'))
            print("Run Advanced specified: "+str(data.runAdvanced))

        # Get the PM task
        task = data.task
        # Get the data
        data_df = read_csv_file(data.file.path)
        # Get the col information
        col_type = OrderedDict()
        features = self.get_features(data)
        for f in features:
            col_type[f.feature] = f.type

        # Get All samples
        # samples = self.get_samples(data)

        # Run Predictive Model (data_df = transformed data_df)
        models_df, train_h2o_df, categorical_index = run_modelMiner_training(data_df, col_type, number_of_model=data.nModel, max_runtime=data.maxRuntime)
        print("Predictive Model Completed!")
        print("Time Taken: ", (datetime.datetime.now() - time_now).total_seconds())
        time_now = datetime.datetime.now()
        print(models_df.columns)

        # Store the Predicive Model
        for i,model in enumerate(models_df.index.values):
            pmml = PMMachineLearning(uid=uid, name=models_df.iloc[i]['model_id'])
            if 'auc' in models_df.columns:
                pmml.auc = models_df.iloc[i]['auc']
                pmml.accuracy = models_df.iloc[i]['auc']
            pmml.logloss = models_df.iloc[i]['logloss']
            pmml.mean_per_class_error = models_df.iloc[i]['mean_per_class_error']
            pmml.rmse = models_df.iloc[i]['rmse']
            pmml.mse = models_df.iloc[i]['mse']
            # Save the model as file
            mfile = "static/file/"+ models_df.iloc[i]['model_id'] 
            pmml.modelfile = mfile # File(open(mfile, 'rb'))
            pmml.parameter = models_df.iloc[i]['param']
            pmml.fullParameter = models_df.iloc[i]['fparam']
            pmml.summary = models_df.iloc[i]['summary']
            numberCV = len(models_df.iloc[i]['cv_name'])
            pmml.numberCV = numberCV # Get the #cv
            
            # Get the file path to the model
            model_path = pmml.modelfile.path
            # Run Model Prediction - and CV_predictions
            prediction_df, mcc, f1score, cv_prediction_arr = run_model_prediction(train_h2o_df, model_path, models_df.iloc[i]['cv_path'])
            
            pmml.mcc = mcc
            pmml.f1score = f1score
            pmml.save()
            

            #####
            # Save each CV model
            #####
            pmmlcv_arr = []
            for j,cvname in enumerate(models_df.iloc[i]['cv_name']):
                cvfile = "static/file/"+ cvname
                pmmlcv_model = PMMLCV(ml=pmml, name=cvname, modelfile=cvfile)
                pmmlcv_arr.append(pmmlcv_model)
            if len(pmmlcv_arr)>0:
                PMMLCV.objects.bulk_create(pmmlcv_arr)

            #####
            # Save the model prediction
            #####
            PMMLPrediction_arr = []
            # Save the model prediction for each sample
            for j,sname in enumerate(data_df.index.values):
                sample = self.get_sample(data, sname)
                if sample is not None and len(prediction_df['predict']) >= j:
                    pred_value = prediction_df['predict'][j]
                    # Get the confidence: prediction vs actual
                    confidence = 0.0
                    if str(pred_value) == str(data_df[data_df.columns[0]][j]):
                        confidence = 100.0
                    pmml_pred = PMMLPrediction(ml=pmml, sample=sample, value=pred_value, confidence=confidence)
                    PMMLPrediction_arr.append(pmml_pred)
            if len(PMMLPrediction_arr)>0:
                PMMLPrediction.objects.bulk_create(PMMLPrediction_arr)
            
            print("------------------------------------------------------------------------------------------------------------------")
            print("Model "+str(pmml.name)+", " +str(pmml.auc)+ "(auc): Saved!")

        print("Top "+str(data.nModel)+" Predictive Model Saved!")

        print(datetime.datetime.now())
        after = datetime.datetime.now()
        difference = (after - before).total_seconds()
        print("Time Taken: ", difference, " seconds")
        serializer = PredictiveModelDetailSerializer(data)
        return Response(serializer.data) 


# /pm/uid - Display PM result detail
class PMAPIViewDetail(APIView):
    renderer_classes = (JSONRenderer, )
    def get_object(self, uid):
        try:
            return PredictiveModel.objects.get(pk=uid)
        except PredictiveModel.DoesNotExist:
            return None

    def get(self, request, uid, format=None):
        data = self.get_object(uid)
        serializer = PredictiveModelDetailSerializer(data)
        return Response(serializer.data) 


# Predictive Model - End 


        

    
