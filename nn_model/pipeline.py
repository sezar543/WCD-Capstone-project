# from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
# from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
# from feature_engine.selection import DropFeatures
# from feature_engine.transformation import LogTransformer
# from feature_engine.wrappers import SklearnTransformerWrapper
# from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from nn_model.config.core import config
from nn_model.processing.nn_architecture import nn_architecture
# from nn_model.processing import features as pp

nn_pipe = Pipeline(
    [
        ("nn_architecture", nn_architecture(variables=[], reference_variable="")),
    ]
)
