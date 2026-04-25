import optuna
import joblib
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def select_best_regressor(x_train, y_train, x_val, y_val):
    print("--- [1단계] 회귀 모델 선택 (기본 파라미터) ---")
    models = {
        # "RandomForest": RandomForestRegressor(random_state=42),
        # "GradientBoosting": GradientBoostingRegressor(random_state=42),
        # "XGBoost": XGBRegressor(random_state=42),
        # "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
        "Ridge": Ridge(random_state=42),
        # "ElasticNet": ElasticNet(random_state=42),
        # "SVR": SVR()
    }
    
    best_model_name = None
    best_score = -float('inf')
    
    for name, model in models.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        score = r2_score(y_val, pred)
        print(f"  {name} Validation R2: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_model_name = name
            
    print(f"-> 선택된 최고 성능 회귀 모델: {best_model_name} (R2: {best_score:.4f})\n")
    return best_model_name

def tune_regressor_hyperparameters(model_name, x_train, y_train, x_val, y_val):
    print(f"--- [2단계] {model_name} 하이퍼파라미터 튜닝 ---")
    def objective(trial):
        model = None
        if model_name == "RandomForest":
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            model = RandomForestRegressor(**param)
        elif model_name == "GradientBoosting":
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'random_state': 42
            }
            model = GradientBoostingRegressor(**param)
        elif model_name == "XGBoost":
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'random_state': 42
            }
            model = XGBRegressor(**param)
        elif model_name == "LightGBM":
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'random_state': 42,
                'verbose': -1
            }
            model = LGBMRegressor(**param)
        elif model_name == "Ridge":
            param = {
                'alpha': trial.suggest_float('alpha', 1e-4, 1e2, log=True),
                'random_state': 42
            }
            model = Ridge(**param)
        elif model_name == "ElasticNet":
            param = {
                'alpha': trial.suggest_float('alpha', 1e-4, 1e2, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'random_state': 42
            }
            model = ElasticNet(**param)
        elif model_name == "SVR":
            param = {
                'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'epsilon': trial.suggest_float('epsilon', 1e-3, 1.0, log=True)
            }
            model = SVR(**param)
            
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        return r2_score(y_val, pred)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    print(f"최적 파라미터:", study.best_params)
    print(f"최고 Validation R2: {study.best_value:.4f}")

    best_model = None
    if model_name == "RandomForest":
        best_model = RandomForestRegressor(**study.best_params, random_state=42)
    elif model_name == "GradientBoosting":
        best_model = GradientBoostingRegressor(**study.best_params, random_state=42)
    elif model_name == "XGBoost":
        best_model = XGBRegressor(**study.best_params, random_state=42)
    elif model_name == "LightGBM":
        best_model = LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
    elif model_name == "Ridge":
        best_model = Ridge(**study.best_params, random_state=42)
    elif model_name == "ElasticNet":
        best_model = ElasticNet(**study.best_params, random_state=42)
    elif model_name == "SVR":
        best_model = SVR(**study.best_params)

    best_model.fit(x_train, y_train)
    return best_model

def tune_and_train_regressor(x_train, y_train, x_val, y_val, target_name):
    print(f"\n[{target_name} 예측 모델] 단계별 모델 학습 ===")
    best_model_name = select_best_regressor(x_train, y_train, x_val, y_val)
    best_model = tune_regressor_hyperparameters(best_model_name, x_train, y_train, x_val, y_val)
    return best_model, best_model_name

def train_and_evaluate(df_train, df_val, df_test):
    """
    양산 DES 속도 (mass_des_speed)를 예측하는 회귀 모델 학습 및 평가
    """

    # ==============================================================
    # 회귀 모델 학습: Target은 mass_des_speed
    # ==============================================================
    
    # Feature와 Target 분리
    y_train = df_train['mass_des_speed']
    x_train = df_train.drop(columns=['mass_des_speed'])
    
    y_val = df_val['mass_des_speed']
    x_val = df_val.drop(columns=['mass_des_speed'])
    
    y_test = df_test['mass_des_speed']
    x_test = df_test.drop(columns=['mass_des_speed'])

    model, name = tune_and_train_regressor(x_train, y_train, x_val, y_val, "양산 DES 속도")

    print("\n=== 데이터셋별 양산 DES 속도 회귀 성능 (PCA 피처 적용) ===")
    def evaluate_model_reg(m, x_data, y_target, dataset_name):
        pred = m.predict(x_data)
        mse = mean_squared_error(y_target, pred)
        mae = mean_absolute_error(y_target, pred)
        r2 = r2_score(y_target, pred)
        print(f"[{dataset_name} Set] R2 Score: {r2:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f}")

    evaluate_model_reg(model, x_train, y_train, "Train")
    evaluate_model_reg(model, x_val, y_val, "Validation")

    print("\n=== [최종 테스트] 양산 DES 속도 예측 Test 데이터 평가 ===")
    predicted_speed = model.predict(x_test)
    
    sim_mse = mean_squared_error(y_test, predicted_speed)
    sim_mae = mean_absolute_error(y_test, predicted_speed)
    sim_r2 = r2_score(y_test, predicted_speed)

    print(f"[Test Set] 예측된 양산 DES 속도 평가")
    print(f"R2 Score: {sim_r2:.4f} | MSE: {sim_mse:.4f} | MAE: {sim_mae:.4f}")

    result_df = pd.DataFrame({
        '실제_양산속도': y_test,
        '예측_추천양산속도': predicted_speed
    })
    
    print("\n[테스트 샘플 결과 Top 5]")
    print(result_df.head(5))

    joblib.dump(model, f'best_{name}_mass_speed_regressor.pkl')
    print(f"\n모델 저장 완료 (best_{name}_mass_speed_regressor.pkl)")

    with open('feature_columns_regressor.json', 'w', encoding='utf-8') as f:
        json.dump(x_train.columns.tolist(), f, ensure_ascii=False, indent=4)
    
    return model, None
