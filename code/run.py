from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# from model_nn import ModelNN
from model_xgb import ModelXGB
from runner import Runner
from util import Submission

if __name__ == '__main__':

    with open("../configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    params_xgb = config["params_xgb"]
    params_nn = config["params_nn"]

    # 特徴量の指定
    features = [f'feat_{i}' for i in range(1, 94)]
    current_year = datetime.now().year
    current_day = datetime.now().day

    # xgboostによる学習・予測
    runner = Runner(f'xgb_{current_year}_{current_day}', ModelXGB, features, params_xgb)
    runner.run_train_cv()
    runner.run_predict_cv()
    Submission.create_submission(f'xgb_{current_year}_{current_day}')

    # # ニューラルネットによる学習・予測
    # runner = Runner('nn1', ModelNN, features, params_nn)
    # runner.run_train_cv()
    # runner.run_predict_cv()
    # Submission.create_submission('nn1')

    '''
    # (参考）xgboostによる学習・予測 - 学習データ全体を使う場合
    runner = Runner('xgb1-train-all', ModelXGB, features, params_xgb_all)
    runner.run_train_all()
    runner.run_test_all()
    Submission.create_submission('xgb1-train-all')
    '''
