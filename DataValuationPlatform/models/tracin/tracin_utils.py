#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 16:26:59 2023

@author: joshua

This file contains a function used to predict false positives using TracIn,
as well as assisting functions and models necessary for TracIn to work
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Set
import re
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback
##################################################################################
#Models used for TracIn
##################################################################################


physical_devices = tf.config.list_physical_devices("GPU")
if (len(physical_devices) >0):
    tf.config.experimental.set_visible_devices(physical_devices[0], "GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


#metrics used to fit the model for TracIn
METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


#DNN that showed best performance during Tracin implementation development; heavily regularized using
#kernel regularization and dropout layers

class ProgressCallback(Callback):
    def on_train_begin(self, logs=None):
        self.total_epochs = self.params['epochs']
        print(f'Training for {self.total_epochs} epochs.')

    def on_epoch_end(self, epoch, logs=None):
        completed = (epoch + 1) / self.total_epochs
        sys.stdout.write(f'\rProgress: {completed:.2%}')
        sys.stdout.flush()
        
        
def make_model(metrics:List = METRICS, 
                output_bias: float = None, 
                x_train: np.array = None,
                random_state:int =0,
                )-> tf.keras.Model:
    """
    

    Parameters
    ----------
    metrics : List[keras.metrics], optional
        Metrics that the model is evaluated upon. The default is METRICS.
    output_bias : float, optional
        Can be set to initial bias, as in the imbalance of the dataset.
        The default is None.
    x_train : np.array, optional
        x_train should be fed to allow the model to know the input shape.
        If left out, the first layer has to be adjusted to not specify the
        input shape. The default is None.
    random_state : int, optional
        Seed to use as random seed for model creation. The default is 0.

    Returns
    -------
    model : Sequential
        returns a highly regularized sequential model using elu and sigmoid
        as activation functions.

    """
    #set the random seed for this session, allowing reproducibility
    #and options for replicates to be different if need be
    tf.random.set_seed(random_state)
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    model = tf.keras.Sequential(
        [
    layers.Dense(512, input_shape=(x_train.shape[-1],), 
                 kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
         metrics=metrics)
 
    return model



def create_lstm_model(input_shape : Tuple[int,int],
                      metrics : List = METRICS,
                      output_bias: float = None,
                      random_state:int = 0
                      )-> tf.keras.Model :
    """
    

    Parameters
    ----------
    input_shape : Tuple[int,int]
        Shape of each one hot encoded sample: should be :(max_token_length, charset_length)
    metrics : list[tf.keras.metrics], optional
        DESCRIPTION. The default is METRICS.
    output_bias : float, optional
        DESCRIPTION. The default is None.
    random_state : int, optional
        Seed to use as random seed for model creation. The default is 0.

    Returns
    -------
    model : Sequential
        LSTM model that can handle sequence data (here strings).

    """
    #set the random seed for this session, allowing reproducibility
    #and options for replicates to be different if need be
    tf.random.set_seed(random_state)
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, dropout=0.2))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid', bias_initializer=output_bias))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics
    )
    return model


##################################################################################
#Functions used to vectorize Smiles when using LSTM model
#################################################################################

def tokenize_smiles(smi: str
                    )-> List[str] :
    """
    This function uses the pattern shown below to tokenize input smiles into
    

    Parameters
    ----------
    smi : str
        Smiles representation of a molecule

    Returns
    -------
    tokens : list[str]
        Returns a list of the tokens that smi is made off

    """
    #pattern of all possible smiles characters used for smiles tokenization
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens


def get_charset(tokenized_smiles: List
                )-> Set[str]:
    """
    

    Parameters
    ----------
    tokenized_smiles : List
        List of tokenized smiles. Preferably a concatenation of train and val
        smiles. These could then be used to generate a charset that encompasses
        all chars necessary for that dataset

    Returns
    -------
    Set[str]
        returns the char set encompassing all chars present in the dataset 

    """
    return set(token for tokens in tokenized_smiles for token in tokens)


def vectorize_smiles(train: pd.DataFrame,
                     val: pd.DataFrame,
                     max_length: int = 120
                     )-> Tuple[np.array, np.array, pd.DataFrame, pd.DataFrame, Dict]:
   
    """
     

     Parameters
     ----------
     train : pd.DataFrame
         Dataframe of shape (M,5) gotten by loading in the train.csv file
         containing the smiles strings of the samples, as well as their 
         primary and confirmatory labels and raw scores
     val : pd.DataFrame
         Dataframe of shape (N,5) gotten by loading in the val.csv file
         containing the smiles strings of the samples, as well as their 
         primary and confirmatory labels and raw scores
     max_length : int, optional
         max_length of the tokenized smiles. The default is 120.

     Returns
     -------
     feature_matrix_train : np.array(A,B,C)
         Feature matrix of the train samples with Dimension A being number
         of samples (if none were longer than max_length, A=M), Dimension
         B being max_length (if default:120), Dimension C being length of the
         charset
     feature_matrix_val : np.array(D,B,C)
         Feature matrix of the val samples with Dimension D being number
         of samples (if none were longer than max_length, D=N), Dimension
         B being max_length (if default:120), Dimension C being length of the
         charset
     db_train : pd.DataFrame
         same as input dataframe, but potentially with less rows if samples were
         removed to to len(sample)>max_length
     db_val : TYPE
         same as input dataframe, but potentially with less rows if samples were
         removed to to len(sample)>max_length
     char_to_int : TYPE
         Dictionary usable to translate vectorized smiles back to strin smiles

     """
    db_train = train.copy()
    db_val = val.copy()
    #creates tokenized smiles for train and val set, as well as 1 set containing 
    #all samples to create one complete charset 
    tokenized_smiles_train = [tokenize_smiles(smi) for smi in db_train["SMILES"]]
    tokenized_smiles_val = [tokenize_smiles(smi) for smi in db_val["SMILES"]]
    tokenized_smiles_all = tokenized_smiles_train + tokenized_smiles_val
    #charset encompassing all possible tokens
    charset = get_charset(tokenized_smiles_all)
    
    final_tokens_train = []
    final_tokens_val = []
    if max_length == None:
        max_length = max([len(tokens) for tokens in tokenized_smiles_all])
    
    #create tokenized smiles and drop samples if tokenized representation is
    #longer than max length
    for i in range (len(tokenized_smiles_train)):
        if len(tokenized_smiles_train[i]) <= max_length:
            final_tokens_train.append(tokenized_smiles_train[i])
        else:
            final_tokens_train.append(tokenized_smiles_train[i][:max_length])
            
            
    for j in range(len(tokenized_smiles_val)):
        if len(tokenized_smiles_val[j]) <= max_length:
            final_tokens_val.append(tokenized_smiles_val[j])
        else:
            db_val.drop([j], inplace=True)
            
    #vectorize tokenized smiles, generating 3D feature matrix,
    #first dimension being the number of samples, second dimension being the 
    #max length of one sample, third dimension being the length of the charset
    #each sample is saved as a 2d array with 1 entry per token, max_length number
    #of entries total
    
    char_to_int = {c: i for i, c in enumerate(charset)}
    feature_matrix_train = np.zeros((len(final_tokens_train), max_length, len(charset)), dtype=np.int8)
    for i, tokens in enumerate(final_tokens_train):
        for j, token in enumerate(tokens):
            feature_matrix_train[i, j, char_to_int[token]] = 1
            
    feature_matrix_val = np.zeros((len(final_tokens_val), max_length, len(charset)), dtype=np.int8)
    for i, tokens in enumerate(final_tokens_val):
        for j, token in enumerate(tokens):
            feature_matrix_val[i, j, char_to_int[token]] = 1
    
    db_train.reset_index(drop=True, inplace=True)
    db_val.reset_index(drop=True, inplace=True)
    return feature_matrix_train, feature_matrix_val, db_train, db_val, char_to_int


##################################################################################
#Functions used for TracIn
##################################################################################



@tf.function
def run_test_influence(inputs_train: Tuple[tf.Tensor, tf.Tensor], 
                        val_features: tf.Tensor, 
                        val_labels: tf.Tensor, 
                        models: List[tf.keras.Model]
                        )-> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    this function calculates the influence scores of all training samples by
    multiplying the gradients of each training sample with the gradients of all
    validation samples. The results are then summed over all weights and averaged 
    over all validation samples.
    Finally, the scores are summed for each training sample over all checkpoints
    using @tf.function speeds the function up by a factor of ~60.

    Parameters
    ----------
    inputs_train : (tf.Tensor(Batch_size, num_features)dtype=float32, tf.Tensor(Batch_size,)dtype=int64)
        Batch of Dataset containing 1 tf.Tensor with the training features and
        1 tf.Tensor with the training labels.
    val_features : tf.Tensor(num_val_samples, num_features)dtype=float32
        Tensor containing all features of the validation set.
    val_labels : tf.Tensor(num_val_samples,)dtype=float64
        Tensor containing the validation set labels.
    models : List[tensorflow.python.keras.engine.sequential.Sequential]
        List of models with loaded weights according to the chosen checkpoints.

    Returns
    -------
    test_influences : tf.Tensor(Batch_size,),dtype=float32
        Returns test_influence scores for each training sample in the Batch.
    train_labels : tf.Tensor(Batch_size,1),dtype=int64
        Train labels of the train samples in the batch.
    probs_train : tf.Tensor(Batch_size,1),dtype=float32
        probabilities of the train samples
    predicted_labels : tf.Tensor(Batch_size,1),dtype=int32
        Predicted labels of the train samples.

    """
    #inputs train is the training batch that contains the features and labels
    train_features, train_labels = inputs_train
    test_influences = []
    #each model m represents one checkpoint of the original model
    for m in models:
        #calculation of the gradients at this CP of the training samples
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            #the last 3 layers are analyzed as analyzing all layers is far too 
            #complex. In fact, one could also only analyze the second to last 
            #layer, as the other two layers are dropout/output layer that don't
            #really contribute to the influence.
            tape.watch(m.trainable_weights[-3:])
            #probabilities of the training samples are calculated
            probs_train = m(train_features)
            
            #make sure train_labels and probs have same shape
            train_labels = tf.reshape(train_labels, [len(train_labels), 1])
            
            #loss of the training samples is calculated
            #had to change from sparse categorical cross entropy to binary
            #crossentropy due to the binary classification application
            loss_train = tf.keras.metrics.binary_crossentropy(train_labels, probs_train)
            
        #the gradients are calculated via the jacobian, that represents the 
        #gradients of a vector valued function. Each row contains the gradient
        #of on of the vectors elements (elementwhise partial derivatves)
        grads_train = tape.jacobian(loss_train, m.trainable_weights[-3:])   
        
        #calculation of the gradients at this CP of the validation samples
        with tf.GradientTape(watch_accessed_variables=False) as tape:
                
                tape.watch(m.trainable_weights[-3:])
                probs_val = m(val_features)
                val_labels = tf.reshape(val_labels, [len(val_labels), 1])                
                loss_val = tf.keras.metrics.binary_crossentropy(val_labels, probs_val)
        grads_val = tape.jacobian(loss_val, m.trainable_weights[-3:])

        results = []
        
        #gradient_train and gradient_val contain gradients for each of the 3
        #layers. Each iteration of this loop calculates the influence scores
        #for one layer and adds them to the overall influence scores
        for gradient_train, gradient_val in zip(grads_train, grads_val):
            # Remove the 3rd dimension (size 1) if dimension == (x,y,1)
            # do it via if-statement because final layer has dimension (x,1),
            #which should not be squeezed
            if gradient_train.shape.ndims == 3:
                gradient_train = tf.squeeze(gradient_train, axis=-1)
            if gradient_val.shape.ndims == 3:
                gradient_val = tf.squeeze(gradient_val, axis=-1)

            #multiply gradients of each training sample with all val gradients
            #resulting matrix: (x,y,z),
            #x = number of training samples
            #y = number of validation samples
            #z = number of weights/nodes
            C = gradient_train[:, tf.newaxis, :] * gradient_val
            #sum all results over all weights for each validation sample for 
            #each training sample
            #resulting matrix: (x,y)
            C = tf.reduce_sum(C, axis=2)
            #average for each training sample over all validation sample
            #resulting vector: (x,)
            C = tf.reduce_mean(C, axis=1)
            #the results are temporarily saved in C
            results.append(C)
        
        #influence scores of each training sample are summed element-whise 
        #over each checkpoint and each layer       
        test_influences = tf.add_n(results)

  # Using probs from last checkpoint
    probs_train, predicted_labels = tf.math.top_k(probs_train, k=1)
    return test_influences, train_labels, probs_train, predicted_labels



#function that implements the batchwhise application of run_test_influence and
#concatenates the results to return one coherent list for all training samples 
def get_test_influence(ds: tf.data.Dataset, 
                        val_features: np.array,
                        val_labels: np.array,
                        models: List[tf.keras.Model]
                        )-> Dict[str, np.array]:
    """
    Parameters
    ----------
    ds : BatchDataset
        Batched dataset that can be iterated through. Made via
        tf.data.Dataset.from_tensor_slices((x_train, train_labels)).batch(Batch_size)
    val_features : np.array(num_val_samples, num_features)
        Validation set features.
    val_labels : np.array(num_val_samples,)
        Validation set labels.
    models : list[tensorflow.python.keras.engine.sequential.Sequential]
        List of models with loaded weights according to the chosen checkpoints.

    Returns
    -------
    dict
        Returns dictionary of concatenated outputs from run_test_influence.

    """
    test_influences_np = []
    labels_np = []
    probs_np = []
    predicted_labels_np = []
    #for each batch d in the dataset ds
    for d in ds:
        test_influences, labels, probs, predicted_labels = run_test_influence(d, val_features, val_labels, models)  
        test_influences_np.append(test_influences.numpy())
        labels_np.append(labels.numpy())
        probs_np.append(probs.numpy())
        predicted_labels_np.append(predicted_labels.numpy()) 
    return {'test_influences': np.concatenate(test_influences_np),
          'labels': np.concatenate(labels_np),
          'probs': np.concatenate(probs_np),
          'predicted_labels': np.concatenate(predicted_labels_np)
         }    
#-----------------------------------------------------------------------------#

@tf.function
def run_self_influence(inputs: Tuple[tf.Tensor, tf.Tensor], 
                       models: List[tf.keras.Model]
                       )->Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Parameters
    ----------
    inputs_train : (tf.Tensor(Batch_size, num_features)dtype=float32, tf.Tensor(Batch_size,)dtype=int64)
        Batch of Dataset containing 1 tf.Tensor with the training features and
        1 tf.Tensor with the training labels.
    models : List[tensorflow.python.keras.engine.sequential.Sequential]
        List of models with loaded weights according to the chosen checkpoints.


    Returns
    -------
    self_influences : tf.Tensor(Batch_size,),dtype=float32
        Returns test_influence scores for each training sample in the Batch.
    train_labels : tf.Tensor(Batch_size,1),dtype=int64
        Train labels of the train samples in the batch.
    probs : tf.Tensor(Batch_size,1),dtype=float32
        probabilities of the train samples
    predicted_labels : tf.Tensor(Batch_size,1),dtype=int32
        Predicted labels of the train samples.

    """
    train_features, train_labels = inputs
    self_influences = []
    for m in models:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            #which layer is being looked at
            #--> maybe play around with which layer to analyse w TracIn
            tape.watch(m.trainable_weights[-2:])
            probs = m(train_features)
            #make sure train_labels and probs have same shape
            train_labels = tf.reshape(train_labels, [len(train_labels), 1])
            #had to change from sparse categorical cross entropy to binary 
            loss = tf.keras.metrics.binary_crossentropy(train_labels, probs)
        grads = tape.jacobian(loss, m.trainable_weights[-2:])
        scores = tf.add_n([tf.math.reduce_sum(
            grad * grad, axis=tf.range(1, tf.rank(grad), 1)) 
            for grad in grads])
        self_influences.append(scores)  

  # Using probs from last checkpoint
    probs, predicted_labels = tf.math.top_k(probs, k=1)
    return tf.math.reduce_sum(tf.stack(self_influences, axis=-1), axis=-1), train_labels, probs, predicted_labels


def get_self_influence(ds: tf.data.Dataset, 
                       models: List[tf.keras.Model]
                       )->Dict[str, np.array]:
    """
    Parameters
    ----------
    ds : BatchDataset
        Batched dataset that can be iterated through. Made via
        tf.data.Dataset.from_tensor_slices((x_train, train_labels)).batch(Batch_size)
    models : list[tensorflow.python.keras.engine.sequential.Sequential]
        List of models with loaded weights according to the chosen checkpoints.

    Returns
    -------
    dict
        Returns dictionary of concatenated outputs from run_test_influence.

    """
    self_influences_np = []
    labels_np = []
    probs_np = []
    predicted_labels_np = []
    for d in ds:
        #d is one batch in the dataset
        self_influences, labels, probs, predicted_labels = run_self_influence(d, models)  
        self_influences_np.append(self_influences.numpy())
        labels_np.append(labels.numpy())
        probs_np.append(probs.numpy())
        predicted_labels_np.append(predicted_labels.numpy()) 
    return {'self_influences': np.concatenate(self_influences_np),
          'labels': np.concatenate(labels_np),
          'probs': np.concatenate(probs_np),
          'predicted_labels': np.concatenate(predicted_labels_np)
         }    


    


