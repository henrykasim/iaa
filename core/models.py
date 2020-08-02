from django.db import models
from django.db.models.signals import pre_delete
from django.dispatch import receiver
from datetime import datetime 

# Predictive Model
class PredictiveModel(models.Model):
    uid = models.CharField(max_length=16,unique=True, primary_key=True)
    model = models.CharField(max_length=255,default="")
    task = models.CharField(max_length=32, default="") # regression or classification
    file = models.FileField(blank=True, null=True, upload_to='static/file/')
    nModel = models.PositiveSmallIntegerField(default="1")
    maxRuntime = models.PositiveIntegerField(default="60") # Default runtime to run automl in seconds
    
# Store the selected features only (might be duplicate from selectedFeatures)
class PredictiveModelFeature(models.Model):
    pm = models.ForeignKey(PredictiveModel, on_delete=models.CASCADE, related_name='features')
    feature = models.CharField(max_length=255)
    type = models.CharField(max_length=32, default='categorical') # categorical or numerical

# Store the samples (train & test)
class PredictiveModelSample(models.Model):
    pm = models.ForeignKey(PredictiveModel, on_delete=models.CASCADE, related_name='samples')
    sample = models.CharField(max_length=255)
    type = models.CharField(max_length=32, default='train') # train or test
    target = models.CharField(max_length=32, default=" ")

# Store the top predictive models
class PMMachineLearning(models.Model):
    uid = models.CharField(max_length=16, default="")
    name = models.CharField(max_length=255, default="")
    keep = models.PositiveSmallIntegerField(default="1") # Not used: If the user wants to keep this Machine Learning Model (1) or not (0)
    rank = models.PositiveSmallIntegerField(default="0") # Not used: model ranking
    recommendation = models.PositiveSmallIntegerField(default="0") # Not used: for recommendation if any
    accuracy = models.FloatField(default="0")
    auc	= models.FloatField(default="0")
    logloss = models.FloatField(default="0")
    mean_per_class_error = models.FloatField(default="0")
    rmse = models.FloatField(default="0")
    mse = models.FloatField(default="0")
    modelfile = models.FileField(blank=True, null=True, upload_to='static/file/')
    summary = models.TextField(default="")
    parameter = models.TextField(default="")
    fullParameter = models.TextField(default="")
    mcc = models.FloatField(default="0")
    f1score = models.FloatField(default="0")
    numberCV = models.PositiveSmallIntegerField(default="1")

# Machine Learning for storing - Cross Validation Predictive Model
class PMMLCV(models.Model):
    ml = models.ForeignKey(PMMachineLearning, on_delete=models.CASCADE, related_name='cvs')
    name = models.CharField(max_length=255, default="")
    modelfile = models.FileField(blank=True, null=True, upload_to='static/file/')

# To store PMML: Sample (train) and Sample(test) predicted value
class PMMLPrediction(models.Model):
    ml = models.ForeignKey(PMMachineLearning, on_delete=models.CASCADE, related_name='predictions')
    sample = models.ForeignKey(PredictiveModelSample, on_delete=models.CASCADE, related_name='sample_predictions')
    value = models.CharField(max_length=255, default="")
    confidence = models.FloatField(default="0")
    
