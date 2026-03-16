import pickle
import pandas as pd
class PredictPipeline:
    def __init__(self):
        self.model_path="artifacts/model.pkl"
        self.preprocessor_path="artifacts/preprocessor.pkl"
    def predict(self,features):
        model=pickle.load(open(self.model_path,"rb"))
        self.preprocessor=pickle.load(open(self.preprocessor_path,"rb"))
        data_scaled=self.preprocessor.transform(features)
        prediction=model.predict(data_scaled)
        return prediction
    