import numpy as np
import pandas as pd
import torch
from carla import Data, MLModel

from LiCE.data.Features import Binary, Categorical, Contiguous


class Enc:
    def __init__(self, colmap):
        self.colmap = colmap

    def get_feature_names(self, list):
        ret = []
        for c in list:
            ret += self.colmap[c]
        return ret


# Custom data set implementations need to inherit from the Data interface
class MixedPolytopeDataset(Data):
    def __init__(self, data_handler, data_train, data_test, one_hot_cols):
        # The data set can e.g. be loaded in the constructor
        # self._mapping = mapping
        self.dhandler = data_handler
        self.data_train = data_train
        self.data_test = data_test
        self._target = data_handler.target_feature.name
        self._immutable = []
        self._continuous = []
        self._categorical = []
        self._columns = []
        for f in data_handler.features:
            self._columns.append(f.name)
            if not f.modifiable and isinstance(f, Contiguous):
                self._immutable.append(f.name)
            if isinstance(f, Categorical) or isinstance(f, Binary):
                self._categorical.append(f.name)
            else:
                self._continuous.append(f.name)

        self._columns_transformed = one_hot_cols

        # needed for the CCHVAE
        colmap = {}
        for f in data_handler.features:
            # for c in self._columns:
            colmap[f.name] = list(
                filter(lambda name: f.name in name, self._columns_transformed)
            )
            # colmap[c] = list(filter(lambda name: c in name, self._columns_transformed))
            # if c in self._immutable:
            if not f.modifiable:
                self._immutable += colmap[f.name]

        self.encoder = Enc(colmap)

    # List of all categorical features
    @property
    def categorical(self):
        return self._categorical

    # List of all continuous features
    @property
    def continuous(self):
        return self._continuous

    # List of all immutable features which
    # should not be changed by the recourse method
    @property
    def immutables(self):
        return self._immutable

    # Feature name of the target column
    @property
    def target(self):
        return self._target

    # The full dataset
    @property
    def df(self):
        return pd.concat([self.df_train, self.df_test])

    # The training split of the dataset
    @property
    def df_train(self):
        # return self._encoder.get_encoded_data()
        return self.transform(
            pd.DataFrame(self.data_train, columns=self._columns + [self._target])
        )

    # The test split of the dataset
    @property
    def df_test(self):
        return self.transform(
            pd.DataFrame(self.data_test, columns=self._columns + [self._target])
        )
        # raise NotImplementedError("Test split not contained in the object")

    # Data transformation, for example normalization of continuous features
    # and encoding of categorical features
    def transform(self, df):
        # vals = [self._encoder.encode_datapoint(row) for _, row in df.iterrows()]
        if df.shape[0] == self.dhandler.n_features + 1:
            vals = self.dhandler.encode(df.drop(columns=self._target))
            vals_y = self.dhandler.encode_y(df[self._target])
            result = pd.DataFrame(
                np.concatenate([vals, vals_y], axis=1),
                columns=self._columns_transformed,
            )
            return result
        vals = self.dhandler.encode(df)
        result = pd.DataFrame(vals, columns=self._columns_transformed)
        return result

    # Inverts transform operation
    def inverse_transform(self, df):
        if df.shape[0] == self.dhandler.encoding_width(True) + 1:
            vals = self.dhandler.decode(df.drop(columns=self._target))
            vals_y = self.dhandler.decode_y(df[self._target])
            result = pd.DataFrame(
                np.concatenate([vals, vals_y], axis=1),
                columns=self._columns_transformed,
            )
            return result
        return self.dhandler.decode(df)


# Custom black-box models need to inherit from
# the MLModel interface
class MyNNModel(MLModel):
    def __init__(self, data, model):
        super().__init__(data)
        # The constructor can be used to load or build an
        # arbitrary black-box-model
        self._mymodel = model
        self._featurelist = data._columns_transformed

    # List of the feature order the ml model was trained on
    @property
    def feature_input_order(self):
        return self._featurelist

    # The ML framework the model was trained on
    @property
    def backend(self):
        return "pytorch"

    # The black-box model object
    @property
    def raw_model(self):
        return self._mymodel.model

    # The predict function outputs
    # the continuous prediction of the model
    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        res = self._mymodel.predict(x)  # there is no sigmoid activation at the end
        return int(res > 0)

    # The predict_proba method outputs
    # the prediction as class probabilities
    def predict_proba(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()
        res = self._mymodel.predict(x)
        prob = torch.sigmoid(torch.tensor(res))
        # print(prob)
        return torch.cat([1 - prob, prob], axis=1)  # for Face and CCHVAE
