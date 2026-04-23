import optuna
import joblib
import json
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def train_and_evaluate(x_train, x_test, y_train, y_test):
    """
    Optuna를 활용하여 여러 모델을 비교 및 하이퍼파라미터 튜닝을 진행하고,
    가장 성능이 좋은 모델로 학습 및 평가를 수행합니다.
    """
    print("\n=== Optuna 모델 선택 및 하이퍼파라미터 튜닝 ===")

    def objective(trial):
        regressor_name = trial.suggest_categorical("regressor", ["RandomForest", "GradientBoosting", "XGBoost", "LightGBM"])
        
        if regressor_name == "RandomForest":
            param = {
                'n_estimators': trial.suggest_int('rf_n_estimators', 50, 300),
                'max_depth': trial.suggest_int('rf_max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
                'random_state': 42
            }
            model = RandomForestRegressor(**param)
        elif regressor_name == "GradientBoosting":
            param = {
                'n_estimators': trial.suggest_int('gb_n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('gb_max_depth', 3, 15),
                'random_state': 42
            }
            model = GradientBoostingRegressor(**param)
        elif regressor_name == "XGBoost":
            param = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 15),
                'random_state': 42
            }
            model = XGBRegressor(**param)
        elif regressor_name == "LightGBM":
            param = {
                'n_estimators': trial.suggest_int('lgbm_n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('lgbm_max_depth', 3, 15),
                'random_state': 42,
                'verbose': -1
            }
            model = LGBMRegressor(**param)
        
        # 3-Fold CV
        score = cross_val_score(model, x_train, y_train, cv=3, scoring='r2', n_jobs=-1).mean()
        return score

    # 최적화 수행
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    print("\n=== Optuna 튜닝 결과 ===")
    print("최적 모델 및 파라미터:", study.best_params)
    print(f"최고 CV R2: {study.best_value:.4f}")

    # 최적 모델 추출 및 학습 준비
    best_params = study.best_params.copy()
    best_model_name = best_params.pop('regressor')

    # 파라미터 접두사(prefix) 제거
    model_params = {}
    for k, v in best_params.items():
        prefix = k.split('_')[0] + '_'
        param_name = k[len(prefix):]
        model_params[param_name] = v

    if best_model_name == "RandomForest":
        best_model = RandomForestRegressor(**model_params, random_state=42)
    elif best_model_name == "GradientBoosting":
        best_model = GradientBoostingRegressor(**model_params, random_state=42)
    elif best_model_name == "XGBoost":
        best_model = XGBRegressor(**model_params, random_state=42)
    elif best_model_name == "LightGBM":
        best_model = LGBMRegressor(**model_params, random_state=42, verbose=-1)

    best_model.fit(x_train, y_train)

    # Offset 예측 및 최종 타겟(mass_des_speed) 평가
    y_pred_offset = best_model.predict(x_test)
    final_pred_mass_des_speed = x_test['first_des_speed'] + y_pred_offset
    actual_mass_des_speed = x_test['first_des_speed'] + y_test

    best_mse = mean_squared_error(actual_mass_des_speed, final_pred_mass_des_speed)
    best_mae = mean_absolute_error(actual_mass_des_speed, final_pred_mass_des_speed)
    best_r2 = r2_score(actual_mass_des_speed, final_pred_mass_des_speed)

    print("\n=== 최종 mass_des_speed 예측 성능 ===")
    print(f"MSE: {best_mse:.4f}")
    print(f"MAE: {best_mae:.4f}")
    print(f"R2 Score: {best_r2:.4f}")

    # 모델 저장
    model_filename = f'best_{best_model_name}_speed_offset.pkl'
    joblib.dump(best_model, model_filename)
    print(f"\n모델 저장 완료: {model_filename}")

    # Feature Importance 출력
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': x_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values(by='importance', ascending=False)
        print("\n=== 중요 변수 TOP 10 ===")
        print(feature_importance.head(10))

    # 학습된 피처 컬럼들 저장 (추후 추론용)
    feature_columns = x_train.columns.tolist()
    with open('feature_columns.json', 'w', encoding='utf-8') as f:
        json.dump(feature_columns, f, ensure_ascii=False, indent=4)
    print("feature 컬럼 저장 완료: feature_columns.json")
    
    return best_model
