# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data Valuation using Reinforcement Learning (DVRL)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import numpy as np
from sklearn import metrics
import tensorflow.compat.v1 as tf
import tqdm
from DataValuationPlatform.models.dvrl import dvrl_metrics
from tensorflow.keras import layers as contrib_layers


class Dvrl(object):
  """Data Valuation using Reinforcement Learning (DVRL) class.

    Attributes:
      x_train: training feature
      y_train: training labels
      x_valid: validation features
      y_valid: validation labels
      problem: 'regression' or 'classification'
      pred_model: predictive model (object)
      parameters: network parameters such as hidden_dim, iterations,
                  activation function, layer_number, learning rate
      checkpoint_file_name: File name for saving and loading the trained model
      flags: flag for training with stochastic gradient descent (flag_sgd)
             and flag for using pre-trained model (flag_pretrain)
  """

  def __init__(self, x_train, y_train, x_valid, y_valid,
               problem, pred_model, parameters, checkpoint_file_name, flags):
    """Initializes DVRL."""

    # Inputs
    self.x_train = x_train
    self.y_train = y_train
    self.x_valid = x_valid
    self.y_valid = y_valid

    self.problem = problem

    # One-hot encoded labels
    if self.problem == 'classification':
      self.y_train_onehot = \
          np.eye(len(np.unique(y_train)))[y_train.astype(int)]
      self.y_valid_onehot = \
          np.eye(len(np.unique(y_train)))[y_valid.astype(int)]
    elif self.problem == 'regression':
      self.y_train_onehot = np.reshape(y_train, [len(y_train), 1])
      self.y_valid_onehot = np.reshape(y_valid, [len(y_valid), 1])

    # Network parameters
    self.hidden_dim = parameters['hidden_dim']
    self.comb_dim = parameters['comb_dim']
    self.outer_iterations = parameters['iterations']
    self.act_fn = parameters['activation']
    self.layer_number = parameters['layer_number']
    self.batch_size = np.min([parameters['batch_size'], len(x_train[:, 0])])
    self.learning_rate = parameters['learning_rate']

    # Basic parameters
    self.epsilon = 1e-8  # Adds to the log to avoid overflow
    self.threshold = 0.9  # Encourages exploration

    # Flags
    self.flag_sgd = flags['sgd']
    self.flag_pretrain = flags['pretrain']

    # If the pred_model uses stochastic gradient descent (SGD) for training
    if self.flag_sgd:
      self.inner_iterations = parameters['inner_iterations']
      self.batch_size_predictor = np.min([parameters['batch_size_predictor'],
                                          len(x_valid[:, 0])])

    # Checkpoint file name
    self.checkpoint_file_name = checkpoint_file_name

    # Basic parameters
    self.data_dim = len(x_train[0, :])
    self.label_dim = len(self.y_train_onehot[0, :])

    # Training Inputs
    # x_input can be raw input or its encoded representation, e.g. using a
    # pre-trained neural network. Using encoded representation can be beneficial
    # to reduce computational cost for high dimensional inputs, like images.


    #trying to port to tf2
    
    #self.x_input = tf.keras.Input(name="x_input", shape=(None, self.data_dim), dtype=tf.dtypes.float32)
    
# =============================================================================
# 
    self.x_input = None
    self.y_input = None
# 
# =============================================================================
    # Prediction difference
    # y_hat_input is the prediction difference between predictive models
    # trained on the training set and validation set.
    # (adding y_hat_input into data value estimator as the additional input
    # is observed to improve data value estimation quality in some cases)
    self.y_hat_input = None

    # Selection vector
    self.s_input = None

    # Rewards (Reinforcement signal)
    self.reward_input = 0

    # Pred model (Note that any model architecture can be used as the predictor
    # model, either randomly initialized or pre-trained with the training data.
    # The condition for predictor model to have fit (e.g. using certain number
    # of back-propagation iterations) and predict functions as its subfunctions.
    self.pred_model = pred_model

    # Final model
    self.final_model = pred_model

    # With randomly initialized predictor
    if (not self.flag_pretrain) & self.flag_sgd:
      if not os.path.exists('tmp'):
        os.makedirs('tmp')
      pred_model.fit(self.x_train, self.y_train_onehot,
                     batch_size=len(self.x_train), epochs=0)
      # Saves initial randomization
      pred_model.save_weights('tmp/pred_model.h5')
      # With pre-trained model, pre-trained model should be saved as
      # 'tmp/pred_model.h5'

    # Baseline model
    if self.flag_sgd:
      self.ori_model = copy.copy(self.pred_model)
      self.ori_model.load_weights('tmp/pred_model.h5')

      # Trains the model
      self.ori_model.fit(x_train, self.y_train_onehot,
                         batch_size=self.batch_size_predictor,
                         epochs=self.inner_iterations, verbose=False)
    else:
      self.ori_model = copy.copy(self.pred_model)
      self.ori_model.fit(x_train, y_train)

    # Valid baseline model
    if 'summary' in dir(self.pred_model):
      self.val_model = copy.copy(self.pred_model)
      self.val_model.load_weights('tmp/pred_model.h5')

      # Trains the model
      self.val_model.fit(x_valid, self.y_valid_onehot,
                         batch_size=self.batch_size_predictor,
                         epochs=self.inner_iterations, verbose=False)
    else:
      self.val_model = copy.copy(self.pred_model)
      self.val_model.fit(x_valid, y_valid)

  def data_value_evaluator(self):
    """Returns data value evaluator model.

    Here, we assume a simple multi-layer perceptron architecture for the data
    value evaluator model. For data types like tabular, multi-layer perceptron
    is already efficient at extracting the relevant information.
    For high-dimensional data types like images or text,
    it is important to introduce inductive biases to the architecture to
    extract information efficiently. In such cases, there are two options:

    (i) Input the encoded representations (e.g. the last layer activations of
    ResNet for images, or the last layer activations of BERT for  text) and use
    the multi-layer perceptron on top of it. The encoded representations can
    simply come from a pre-trained predictor model using the entire dataset.

    (ii) Modify the data value evaluator model definition below to have the
    appropriate inductive bias (e.g. using convolutional layers for images,
    or attention layers text).

    Returns:
      dve: data value estimations
    """
    x_input = tf.keras.Input(shape=(self.data_dim,), name="x_input")
    y_input = tf.keras.Input(shape=(self.label_dim,), name="y_input")
    y_hat_input = tf.keras.Input(shape=(self.label_dim,), name="y_hat_input")

    # Build the model
    inter_layer = x_input
    for _ in range(int(self.layer_number - 2)):
        inter_layer = tf.keras.layers.Dense(self.hidden_dim, activation=self.act_fn)(inter_layer)
    inter_layer = tf.keras.layers.Dense(self.comb_dim, activation=self.act_fn)(inter_layer)

    # Combines with y_hat
    comb_layer = tf.keras.layers.concatenate([inter_layer, y_hat_input])
    comb_layer = tf.keras.layers.Dense(self.comb_dim, activation=self.act_fn)(comb_layer)
    dve = tf.keras.layers.Dense(1, activation='sigmoid')(comb_layer)

    # Create Keras Model
    model = tf.keras.Model(inputs=[x_input, y_input, y_hat_input], outputs=dve)

    return model

  
  
  def train_dvrl(self, perf_metric):
    """Trains DVRL based on the specified objective function."""

    # Initialize the data value estimator model
    est_data_value = self.data_value_evaluator()
    y_train_valid_pred = self.val_model.predict(self.x_train)

    # Initial calculation of y_pred_diff
    if self.problem == 'classification':
        #y_train_valid_pred_onehot = tf.keras.utils.to_categorical(y_train_valid_pred, num_classes=2)
        y_train_valid_pred_onehot = self.val_model.predict_proba(self.x_train)
        self.y_pred_diff = np.abs(self.y_train_onehot - y_train_valid_pred_onehot)
    elif self.problem == 'regression':
        self.y_pred_diff = np.abs(self.y_train_onehot - y_train_valid_pred) / self.y_train_onehot

    # Compute initial performance on the validation set
    y_valid_hat = self.ori_model.predict_proba(self.x_valid)
    self.valid_perf = self.calculate_performance_metric(perf_metric, self.y_valid, y_valid_hat[:,1])

    # Initialize the placeholder for the reward
    self.reward_input = 0

    # Prepare optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    for _ in tqdm.tqdm(range(self.outer_iterations)):
        # Batch selection
        batch_idx = np.random.permutation(len(self.x_train[:, 0]))[:self.batch_size]
        x_batch = self.x_train[batch_idx, :]
        y_batch = self.y_train[batch_idx]
        y_batch_onehot = self.y_train_onehot[batch_idx]
        y_hat_batch = self.y_pred_diff[batch_idx]
        with tf.GradientTape() as tape:
            # Generate selection probability
            est_dv_curr = est_data_value([x_batch, y_batch_onehot, y_hat_batch], training=True)
           
            self.s_input = np.random.binomial(1, est_dv_curr, est_dv_curr.shape)
            # Compute generator loss
            dve_loss = self.compute_generator_loss(est_dv_curr)

            # Apply gradients
        gradients = tape.gradient(dve_loss, est_data_value.trainable_variables)
        optimizer.apply_gradients(zip(gradients, est_data_value.trainable_variables))

        # Update model and calculate reward
        self.reward_input = self.update_model_and_calculate_reward(self.pred_model, x_batch,y_batch, y_batch_onehot, perf_metric)
# =============================================================================
# 	# Update y_pred_diff after the model is updated
#         y_train_valid_pred = self.val_model.predict(self.x_train)
#         if self.problem == 'classification':
#             y_train_valid_pred = np.column_stack((1 - y_train_valid_pred, y_train_valid_pred)) 	
#             y_pred_diff = np.abs(self.y_train_onehot - y_train_valid_pred)
#         elif self.problem == 'regression':
#             y_pred_diff = np.abs(self.y_train_onehot - y_train_valid_pred) / self.y_train_onehot
# 
# =============================================================================
    # Save trained model
    est_data_value.save_weights(self.checkpoint_file_name)

        # Train final model with the generated data values
    self.train_final_model(est_data_value)
  
  def compute_generator_loss(self, est_dv_curr):
    # Calculate generator loss (REINFORCE algorithm)
    prob = tf.reduce_sum(self.s_input * tf.math.log(est_dv_curr + self.epsilon) +
                         (1 - self.s_input) * tf.math.log(1 - est_dv_curr + self.epsilon))
    dve_loss = (-self.reward_input * prob) + \
                1e3 * (tf.maximum(tf.reduce_mean(est_dv_curr) - self.threshold, 0) +
                       tf.maximum((1 - self.threshold) - tf.reduce_mean(est_dv_curr), 0))
    return dve_loss  

  def calculate_performance_metric(self, metric, y_true, y_pred):
        # Implement performance metric calculation
        if metric == 'auc':
            return metrics.roc_auc_score(y_true, y_pred)
    
  def update_model_and_calculate_reward(self, model, x_batch, y_batch,y_batch_onehot, perf_metric):
    # Train the model with the current batch
    if 'summary' in dir(model):
        # For neural networks, use inner_iterations for the number of epochs
        model.fit(x_batch, y_batch_onehot, epochs=self.inner_iterations, verbose=False)
    else:
        # For other models, adjust as necessary
        model.fit(x_batch, y_batch) 
    
    # Evaluate performance after training
    y_valid_hat_new = model.predict_proba(self.x_valid)
    dvrl_perf = self.calculate_performance_metric(perf_metric, self.y_valid, y_valid_hat_new[:,1])

    # Calculate reward based on performance change
    if self.problem == 'classification':
      reward_curr = dvrl_perf - self.valid_perf
    elif self.problem == 'regression':
      reward_curr = self.valid_perf - dvrl_perf # valid_perf should be set in train_dvrl before the loop
    #self.valid_perf = dvrl_perf  # Update valid_perf for the next iteration
    return reward_curr


  def train_final_model(self, est_data_value):
    # Generate data values
    y_train_valid_pred = self.val_model.predict(self.x_train)
    if self.problem == 'classification':
        #y_train_valid_pred_onehot = tf.keras.utils.to_categorical(y_train_valid_pred, num_classes=2)
        y_train_valid_pred_onehot = self.val_model.predict_proba(self.x_train)
        self.y_pred_diff = np.abs(self.y_train_onehot - y_train_valid_pred_onehot)
    elif self.problem == 'regression':
        self.y_pred_diff = np.abs(self.y_train_onehot - y_train_valid_pred) / self.y_train_onehot

    final_data_value = est_data_value([self.x_train, self.y_train_onehot, self.y_pred_diff]).numpy()[:, 0]

    # Train the final model using the generated data values
    if 'summary' in dir(self.pred_model):
        self.pred_model.load_weights('tmp/pred_model.h5')
        self.pred_model.fit(self.x_train, self.y_train_onehot,
                            sample_weight=final_data_value,
                            epochs=self.inner_iterations, verbose=False)
    else:
        self.pred_model.fit(self.x_train, self.y_train, sample_weight=final_data_value)

  

  def data_valuator(self, x_train, y_train):
    """Returns data values using the data valuator model.

    Args:
      x_train: training features
      y_train: training labels

    Returns:
      final_dat_value: final data values of the training samples
    """

    # One-hot encoded labels
    if self.problem == 'classification':
      y_train_onehot = np.eye(len(np.unique(y_train)))[y_train.astype(int)]
      #y_train_valid_pred = self.val_model.predict(x_train)
      y_train_valid_pred = self.val_model.predict_proba(x_train)
      
    elif self.problem == 'regression':
      y_train_onehot = np.reshape(y_train, [len(y_train), 1])
      y_train_valid_pred = np.reshape(self.val_model.predict(x_train),
                                      [-1, 1])

    # Generates y_train_hat
    if self.problem == 'classification':
      #y_train_valid_pred_onehot = tf.keras.utils.to_categorical(y_train_valid_pred, num_classes=2)
      y_train_valid_pred_onehot = y_train_valid_pred
      y_train_hat = np.abs(y_train_onehot - y_train_valid_pred_onehot)
    elif self.problem == 'regression':
      y_train_hat = np.abs(y_train_onehot - y_train_valid_pred)/y_train_onehot

    # Load the model weights
    est_data_value = self.data_value_evaluator()
    est_data_value.load_weights(self.checkpoint_file_name)


    final_data_value = est_data_value([x_train, y_train_onehot, y_train_hat]).numpy()[:, 0]

    return final_data_value

  def dvrl_predictor(self, x_test):
    """Returns predictions using the predictor model."""

    if self.flag_sgd:
        y_test_hat = self.final_model.predict(x_test)
    else:
        if self.problem == 'classification':
            # For classification, use predict for TensorFlow 2
            #y_test_hat = self.final_model.predict(x_test)
            y_test_hat = self.final_model.predict_proba(x_test)
        elif self.problem == 'regression':
            y_test_hat = self.final_model.predict(x_test)

    return y_test_hat

