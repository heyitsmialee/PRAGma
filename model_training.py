import optuna
import joblib
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import Ridge, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, classification_report

def _get_suggested_regressor(trial):
    regressor_name = trial.suggest_categorical("regressor", [
        "RandomForest", "GradientBoosting", "XGBoost", "LightGBM", 
        "Ridge", "ElasticNet", "SVR"
    ])
    
    if regressor_name == "RandomForest":
        param = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
            'max_depth': trial.suggest_int('rf_max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            'random_state': 42
        }
        return RandomForestRegressor(**param), regressor_name
        
    elif regressor_name == "GradientBoosting":
        param = {
            'n_estimators': trial.suggest_int('gb_n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('gb_max_depth', 3, 15),
            'random_state': 42
        }
        return GradientBoostingRegressor(**param), regressor_name
        
    elif regressor_name == "XGBoost":
        param = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
            'random_state': 42
        }
        return XGBRegressor(**param), regressor_name
        
    elif regressor_name == "LightGBM":
        param = {
            'n_estimators': trial.suggest_int('lgbm_n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('lgbm_max_depth', 3, 15),
            'random_state': 42,
            'verbose': -1
        }
        return LGBMRegressor(**param), regressor_name
        
    elif regressor_name == "Ridge":
        param = {
            'alpha': trial.suggest_float('ridge_alpha', 1e-4, 1e2, log=True),
            'random_state': 42
        }
        return Ridge(**param), regressor_name
        
    elif regressor_name == "ElasticNet":
        param = {
            'alpha': trial.suggest_float('en_alpha', 1e-4, 1e2, log=True),
            'l1_ratio': trial.suggest_float('en_l1_ratio', 0.0, 1.0),
            'random_state': 42
        }
        return ElasticNet(**param), regressor_name
        
    elif regressor_name == "SVR":
        param = {
            'C': trial.suggest_float('svr_C', 1e-2, 1e2, log=True),
            'gamma': trial.suggest_categorical('svr_gamma', ['scale', 'auto']),
            'epsilon': trial.suggest_float('svr_epsilon', 1e-3, 1.0, log=True)
        }
        return SVR(**param), regressor_name

def _get_suggested_classifier(trial):
    classifier_name = trial.suggest_categorical("classifier", [
        "RandomForest", "GradientBoosting", "XGBoost", "LightGBM", 
        "LogisticRegression", "SVC"
    ])
    
    if classifier_name == "RandomForest":
        param = {
            'n_estimators': trial.suggest_int('rfc_n_estimators', 50, 300),
            'max_depth': trial.suggest_int('rfc_max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('rfc_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rfc_min_samples_leaf', 1, 10),
            'random_state': 42
        }
        return RandomForestClassifier(**param), classifier_name
        
    elif classifier_name == "GradientBoosting":
        param = {
            'n_estimators': trial.suggest_int('gbc_n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('gbc_learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('gbc_max_depth', 3, 15),
            'random_state': 42
        }
        return GradientBoostingClassifier(**param), classifier_name
        
    elif classifier_name == "XGBoost":
        param = {
            'n_estimators': trial.suggest_int('xgbc_n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('xgbc_learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('xgbc_max_depth', 3, 15),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        return XGBClassifier(**param), classifier_name
        
    elif classifier_name == "LightGBM":
        param = {
            'n_estimators': trial.suggest_int('lgbmc_n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('lgbmc_learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('lgbmc_max_depth', 3, 15),
            'random_state': 42,
            'verbose': -1
        }
        return LGBMClassifier(**param), classifier_name
        
    elif classifier_name == "LogisticRegression":
        param = {
            'C': trial.suggest_float('lr_C', 1e-4, 1e2, log=True),
            'random_state': 42,
            'max_iter': 1000
        }
        return LogisticRegression(**param), classifier_name
        
    elif classifier_name == "SVC":
        param = {
            'C': trial.suggest_float('svc_C', 1e-2, 1e2, log=True),
            'gamma': trial.suggest_categorical('svc_gamma', ['scale', 'auto']),
            'probability': True,
            'random_state': 42
        }
        return SVC(**param), classifier_name

def tune_and_train_regressor(x_train, y_train, x_val, y_val, target_name):
    print(f"\n[{target_name} 회귀 예측 모델] Optuna 모델 선택 및 하이퍼파라미터 튜닝 ===")
    def objective(trial):
        model, _ = _get_suggested_regressor(trial)
        model.fit(x_train, y_train)
        pred_val = model.predict(x_val)
        return r2_score(y_val, pred_val)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    print(f"[{target_name}] 최적 모델 및 파라미터:", study.best_params)
    print(f"[{target_name}] 최고 Validation R2: {study.best_value:.4f}")

    best_model, best_model_name = _get_suggested_regressor(study.best_trial)
    best_model.fit(x_train, y_train)
    return best_model, best_model_name

def tune_and_train_classifier(x_train, y_train, x_val, y_val, target_name):
    print(f"\n[{target_name} 분류 예측 모델] Optuna 모델 선택 및 하이퍼파라미터 튜닝 ===")
    def objective(trial):
        model, _ = _get_suggested_classifier(trial)
        model.fit(x_train, y_train)
        pred_val = model.predict(x_val)
        # 불량 여부는 클래스 불균형이 있을 수 있으므로 F1-Macro 사용
        return f1_score(y_val, pred_val, average='macro')

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    print(f"[{target_name}] 최적 모델 및 파라미터:", study.best_params)
    print(f"[{target_name}] 최고 Validation F1(Macro): {study.best_value:.4f}")

    best_model, best_model_name = _get_suggested_classifier(study.best_trial)
    best_model.fit(x_train, y_train)
    return best_model, best_model_name

class SpeedRecommender:
    """
    내부적으로 speed_offset을 예측하지만, 
    최종 아웃풋으로는 (예측된 speed_offset + first_des_speed) 즉, 추천 양산 DES 속도를 반환하는 래퍼 클래스입니다.
    """
    def __init__(self, offset_model):
        self.model = offset_model
        
    def predict(self, x_data):
        # x_data에는 'first_des_speed'가 포함되어 있어야 합니다.
        predicted_offset = self.model.predict(x_data)
        recommended_mass_speed = x_data['first_des_speed'] + predicted_offset
        return recommended_mass_speed

def train_and_evaluate(df_train, df_val, df_test):
    """
    1. 양산DES속도(target_1) 추천용 회귀 모델 학습
    2. 불량여부(target_2) 분류 모델 학습
    3. Test 셋에서 추천 양산DES속도를 도출하고, 그 속도를 적용했을 때의 불량 발생 여부 예측
    """
    # ==============================================================
    # 모델 1: 양산 DES 속도 오프셋(speed_offset) 예측 모델 (회귀)
    # (mass_des_speed, aoi_defect_rate, is_defective 제외)
    # ==============================================================
    drop_cols_1 = ['mass_des_speed', 'speed_offset', 'aoi_defect_rate', 'is_defective']
    x_train_1 = df_train.drop(columns=drop_cols_1, errors='ignore')
    x_val_1   = df_val.drop(columns=drop_cols_1, errors='ignore')
    x_test_1  = df_test.drop(columns=drop_cols_1, errors='ignore')
    
    y_train_1 = df_train['speed_offset']
    y_val_1   = df_val['speed_offset']
    y_test_1  = df_test['speed_offset']

    base_model_1, name_1 = tune_and_train_regressor(x_train_1, y_train_1, x_val_1, y_val_1, "mass_des_speed_offset")
    
    recommender_model = SpeedRecommender(base_model_1)
    
    print("\n=== [모델 1] 데이터셋별 최종 타겟(mass_des_speed) 평가 (회귀) ===")
    def evaluate_model_1(recommender, x_data, df_original, dataset_name):
        final_pred = recommender.predict(x_data)
        actual_target = df_original['mass_des_speed']

        mse = mean_squared_error(actual_target, final_pred)
        mae = mean_absolute_error(actual_target, final_pred)
        r2 = r2_score(actual_target, final_pred)

        print(f"[{dataset_name} Set] R2 Score: {r2:.4f} | MSE: {mse:.4f} | MAE: {mae:.4f}")
        return final_pred

    evaluate_model_1(recommender_model, x_train_1, df_train, "Train")
    evaluate_model_1(recommender_model, x_val_1, df_val, "Validation")
    
    recommended_mass_des_speed = evaluate_model_1(recommender_model, x_test_1, df_test, "Test")

    # ==============================================================
    # 모델 2: 불량 발생 여부(is_defective) 예측 모델 (분류)
    # (실제 mass_des_speed를 feature로 사용, speed_offset, aoi_defect_rate, is_defective는 제외)
    # ==============================================================
    drop_cols_2 = ['speed_offset', 'aoi_defect_rate', 'is_defective']
    x_train_2 = df_train.drop(columns=drop_cols_2, errors='ignore')
    x_val_2   = df_val.drop(columns=drop_cols_2, errors='ignore')
    
    y_train_2 = df_train['is_defective']
    y_val_2   = df_val['is_defective']
    y_test_2  = df_test['is_defective']

    model_2, name_2 = tune_and_train_classifier(x_train_2, y_train_2, x_val_2, y_val_2, "불량 발생 여부(0=정상, 1=불량)")

    print("\n=== [모델 2] 데이터셋별 불량 발생 여부 분류 성능 (실제 속도 기준) ===")
    def evaluate_model_2(model, x_data, y_target, dataset_name):
        pred = model.predict(x_data)
        acc = accuracy_score(y_target, pred)
        f1 = f1_score(y_target, pred, average='macro')
        print(f"[{dataset_name} Set] Accuracy: {acc:.4f} | F1 Score (Macro): {f1:.4f}")
        if dataset_name == "Validation":
            print(f"\n[{dataset_name} Set Classification Report]\n", classification_report(y_target, pred))

    evaluate_model_2(model_2, x_train_2, y_train_2, "Train")
    evaluate_model_2(model_2, x_val_2, y_val_2, "Validation")

    # ==============================================================
    # 최종 시뮬레이션: 추천 양산속도를 적용했을 때의 불량 발생 여부 예측
    # ==============================================================
    print("\n=== [최종 시뮬레이션] 추천 양산속도를 적용한 Test 데이터 평가 ===")
    
    x_test_simulated = df_test.drop(columns=drop_cols_2, errors='ignore').copy()
    x_test_simulated['mass_des_speed'] = recommended_mass_des_speed

    predicted_is_defective = model_2.predict(x_test_simulated)
    
    sim_acc = accuracy_score(y_test_2, predicted_is_defective)
    sim_f1 = f1_score(y_test_2, predicted_is_defective, average='macro')

    print(f"[Test Set 시뮬레이션] 추천 속도 적용 시 예측된 불량 발생 여부 평가")
    print(f"Accuracy: {sim_acc:.4f} | F1 Score (Macro): {sim_f1:.4f}")

    result_df = pd.DataFrame({
        '실제_초물속도': df_test['first_des_speed'],
        '실제_양산속도': df_test['mass_des_speed'],
        '추천_양산속도': recommended_mass_des_speed,
        '실제_불량여부': y_test_2,
        '예측_추천속도적용_불량여부': predicted_is_defective
    })
    
    print("\n[시뮬레이션 샘플 결과 Top 5 (0=정상, 1=불량)]")
    print(result_df.head(5))

    joblib.dump(recommender_model, f'best_{name_1}_speed_recommender.pkl')
    joblib.dump(model_2, f'best_{name_2}_defect_classifier.pkl')
    print(f"\n모델 저장 완료 (best_{name_1}_speed_recommender.pkl, best_{name_2}_defect_classifier.pkl)")

    with open('feature_columns_speed.json', 'w', encoding='utf-8') as f:
        json.dump(x_train_1.columns.tolist(), f, ensure_ascii=False, indent=4)
    with open('feature_columns_classifier.json', 'w', encoding='utf-8') as f:
        json.dump(x_train_2.columns.tolist(), f, ensure_ascii=False, indent=4)
    
    return recommender_model, model_2
