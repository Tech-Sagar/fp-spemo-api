"""MLflow wrapper so the registered model can be loaded with mlflow.pyfunc.load_model(...)
and scored on a DataFrame with a `path` column of audio files."""
import mlflow.pyfunc
import pandas as pd


class MlflowEmotionModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from ser import EmotionModel
        self.model = EmotionModel.load(context.artifacts["ser_model"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        from ser.audio import load_file
        rows = [self.model.predict(load_file(p)) for p in model_input["path"]]
        return pd.DataFrame([{k: v for k, v in r.items() if k != "probabilities"} for r in rows])
