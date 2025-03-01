"""
This module containts helper functions to load data and get meta deta.
"""
import numpy as np
import pandas as pd
import os

import dice_ml

#Dice Imports
from compared_methods.dice_ml_04.dice_ml.utils.sample_architecture.vae_model import CF_VAE

#Pytorch
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

def load_adult_income_dataset():
    """Loads adult income dataset from https://archive.ics.uci.edu/ml/datasets/Adult and prepares the data for data analysis based on https://rpubs.com/H_Zhu/235617

    :return adult_data: returns preprocessed adult income dataset.
    """
    raw_data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', delimiter=', ', dtype=str)

    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    adult_data = pd.DataFrame(raw_data, columns=column_names)


    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

    adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government', 'Local-gov':'Government'}})
    adult_data = adult_data.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

    adult_data = adult_data.replace({'occupation': {'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                                           'Exec-managerial':'White-Collar','Farming-fishing':'Blue-Collar',
                                            'Handlers-cleaners':'Blue-Collar',
                                            'Machine-op-inspct':'Blue-Collar','Other-service':'Service',
                                            'Priv-house-serv':'Service',
                                           'Prof-specialty':'Professional','Protective-serv':'Service',
                                            'Tech-support':'Service',
                                           'Transport-moving':'Blue-Collar','Unknown':'Other/Unknown',
                                            'Armed-Forces':'Other/Unknown','?':'Other/Unknown'}})

    adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married', 'Married-spouse-absent':'Married','Never-married':'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                           'Amer-Indian-Eskimo':'Other'}})

    adult_data = adult_data[['age','workclass','education','marital-status','occupation','race','gender',
                     'hours-per-week','income']]

    adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                           '11th':'School', '10th':'School', '7th-8th':'School', '9th':'School',
                                          '12th':'School', '5th-6th':'School', '1st-4th':'School', 'Preschool':'School'}})

    adult_data = adult_data.rename(columns={'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    return adult_data


def get_adult_income_modelpath(backend='TF1'):
    pkg_path = dice_ml.__path__[0]
    model_ext = '.h5' if 'TF' in backend else '.pth'
    modelpath = os.path.join(pkg_path, 'utils', 'sample_trained_models', 'adult'+model_ext)
    return modelpath

def get_adult_data_info():
    feature_description = {'age':'age',
                        'workclass': 'type of industry (Government, Other/Unknown, Private, Self-Employed)',
                        'education': 'education level (Assoc, Bachelors, Doctorate, HS-grad, Masters, Prof-school, School, Some-college)',
                        'marital_status': 'marital status (Divorced, Married, Separated, Single, Widowed)',
                        'occupation': 'occupation (Blue-Collar, Other/Unknown, Professional, Sales, Service, White-Collar)',
                        'race': 'white or other race?',
                        'gender': 'male or female?',
                        'hours_per_week': 'total work hours per week',
                        'income': '0 (<=50K) vs 1 (>50K)'}
    return feature_description

def get_base_gen_cf_initialization( data_interface, encoded_size, cont_minx, cont_maxx, margin, validity_reg, epochs, wm1, wm2, wm3, learning_rate ):

        # Dataset for training Variational Encoder Decoder model for CF Generation
        # df = data_interface.normalize_data(data_interface.one_hot_encoded_data)
        df = data_interface.one_hot_encoded_data # I include the normalization in that
        encoded_data= df[data_interface.encoded_feature_names + [data_interface.outcome_name]]
        dataset = encoded_data.to_numpy()
        print('Dataset Shape:',  encoded_data.shape)
        print('Datasets Columns:', encoded_data.columns)

        #Normalise_Weights
        normalise_weights={}
        for idx in range(len(cont_minx)):
            _max= cont_maxx[idx]
            _min= cont_minx[idx]
            normalise_weights[idx]=[_min, _max]

        #Train, Val, Test Splits
        np.random.shuffle(dataset)
        test_size= int(data_interface.test_size)
        vae_test_dataset= dataset[:test_size]
        dataset= dataset[test_size:]
        vae_val_dataset= dataset[:test_size]
        vae_train_dataset= dataset[test_size:]

        #BaseGenCF Model
        cf_vae = CF_VAE(data_interface, encoded_size)

        #Optimizer
        cf_vae_optimizer = optim.Adam([
            {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_mean.parameters()),'weight_decay': wm1},
            {'params': filter(lambda p: p.requires_grad, cf_vae.encoder_var.parameters()),'weight_decay': wm2},
            {'params': filter(lambda p: p.requires_grad, cf_vae.decoder_mean.parameters()),'weight_decay': wm3},
            ], lr=learning_rate
        )

        # Check: If base_obj was passsed via reference and it mutable; might not need to have a return value at all
        return vae_train_dataset, vae_val_dataset, vae_test_dataset, normalise_weights, cf_vae, cf_vae_optimizer
