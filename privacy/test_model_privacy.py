seed = 42

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd()+'/configs/'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import numpy as np
np.random.seed(seed)

import logging
import tensorflow as tf
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

import random
random.seed(seed)

from scipy import special
import tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.plotting as plotting
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack import membership_inference_attack as mia
from tensorflow_privacy.privacy.privacy_tests.membership_inference_attack.data_structures import AttackInputData, AttackType, SlicingSpec

tfk = tf.keras


def test_model_privacy(model, X_train, y_train, X_test, y_test):
    
    logits_train = model.predict(X_train, verbose=0)
    logits_test = model.predict(X_test, verbose=0)

    prob_train = special.softmax(logits_train, axis=-1)
    prob_test = special.softmax(logits_test, axis=-1)

    cce = tfk.backend.categorical_crossentropy
    constant = tfk.backend.constant

    loss_train = cce(constant(y_train.astype(int)), constant(prob_train), from_logits=False).numpy()
    loss_test = cce(constant(y_test.astype(int)), constant(prob_test), from_logits=False).numpy()
    
    
    attack_input = AttackInputData(
      logits_train = logits_train,
      logits_test = logits_test,
      loss_train = loss_train,
      loss_test = loss_test,
      labels_train = np.argmax(y_train,axis=1).astype(int),
      labels_test = np.argmax(y_test,axis=1).astype(int)
    )

    slicing_spec = SlicingSpec(
        entire_dataset = True,
        by_classification_correctness = False,
        by_class = False
    )

    attack_types = [
        AttackType.THRESHOLD_ATTACK,
        AttackType.LOGISTIC_REGRESSION,
        AttackType.RANDOM_FOREST
    ] 

    attacks_result = mia.run_attacks(attack_input=attack_input,
                                     slicing_spec=slicing_spec,
                                     attack_types=attack_types,)

    print(attacks_result.get_result_with_max_auc())
    print(plotting.plot_roc_curve(attacks_result.get_result_with_max_auc().roc_curve))
    print(attacks_result.summary(by_slices = True))