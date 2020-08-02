from rest_framework import serializers
# Predictive Model
from core.models import PredictiveModel, PredictiveModelFeature, PredictiveModelSample, PMMachineLearning, PMMLCV, PMMLPrediction

###
# Predictive Model Serializer
###

class PMMLPredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = PMMLPrediction
        fields = '__all__'

class PredictiveModelFeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictiveModelFeature
        fields = ('id', 'feature', 'type')

class PredictiveModelSampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictiveModelSample
        fields = ('id', 'sample', 'type', 'target')

class PMMachineLearningSerializer(serializers.ModelSerializer):
    predictions = PMMLPredictionSerializer(many=True, read_only=True)
    class Meta:
        model = PMMachineLearning
        fields = '__all__'

# Simplified Serializer for PredictiveModel 
class PredictiveModelSerializer(serializers.ModelSerializer):
    features = PredictiveModelFeatureSerializer(many=True, read_only=True)
    samples = PredictiveModelSampleSerializer(many=True, read_only=True)
    class Meta:
        model = PredictiveModel
        fields = '__all__'

# Detailed Serializer for PredictiveModel: including Predictive Model information
class PredictiveModelDetailSerializer(serializers.ModelSerializer):
    models = serializers.SerializerMethodField()
    class Meta:
        model = PredictiveModel
        fields = '__all__'

    def get_models(self, obj):
        # Check if there is any models
        try:
            pmml = PMMachineLearning.objects.filter(uid=obj.uid)
            if pmml:
                return PMMachineLearningSerializer(pmml, many=True, read_only=True).data
        except None:
            pass
        return None
