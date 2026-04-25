import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_preprocess_data(csv_path='./Train_0319.csv', n_components=0.95):
    """
    데이터 로드, 컬럼명 변경, 결측치 처리, 파생변수 생성, 스케일링,
    그리고 차원 축소(PCA)를 수행합니다.
    초물 관련 데이터와 불량률 데이터는 제외하며, 양산 DES 속도를 타겟 변수로 합니다.
    """
    raw_df = pd.read_csv(csv_path, encoding='cp949')
    
    # 초물과 관련된 데이터(초물 DES 속도, Result Ng2 등)와 AOI불량률 제거,
    # 양산DES 속도는 예측해야 할 Target이 되므로 포함.
    df = raw_df[[ 
        # 'Result Ng2', '초물 DES 속도', 'AOI불량률' <- 문제 정의에 따라 제외
        'DRY FILM 정보', '선폭 OFFSET', '양산DES 속도', 
        '거래처', '제품군', '공법구분', 'LAYER', '도금구분', '노광 설비정보', 
        'DES 설비정보', '정면 설비정보', 'Cu 표면두께 Max_Val', 'Cu 표면두께 AVG_VAL', 
        'Cu 표면두께 Min_Val', 'Cu 표면두께 Std_Val', 'Cu 표면두께 Median_Val', 
        '재작업사유', '분석치_Etch factor', '분석치_Etching(염화동) - Cu', 
        '분석치_Etching(염화동) - HCl', '분석치_Etching(염화동) - 비중', 
        '분석치_Etching(염화동) - 온도', '분석치_Etching-첨가제(HB-120EF)', 
        '분석치_Etching량', '분석치_Soft Etch - Cu', '분석치_Soft Etch - H2SO4', 
        '분석치_Soft Etch - SPS', '분석치_박리액 - 농도', '분석치_수세수 - pH', 
        '분석치_현상액 - pH', '분석치_현상액 - 농도'
        # '통합코드' (고유값이 너무 많아 과적합을 유발하므로 제외)
    ]].copy()

    # 영문/단축 컬럼명으로 변경
    new_cols = [
        'df_type', 'width_offset', 'mass_des_speed', 
        'customer', 'product_family', 'process_type', 'layer_count', 
        'plating_type', 'expo_eq_id', 'des_eq_id', 'brush_eq_id', 'cu_thick_max', 
        'cu_thick_avg', 'cu_thick_min', 'cu_thick_std', 'cu_thick_median', 
        'rework_history', 'etch_factor', 'meas_etch_cu', 'meas_etch_hcl', 
        'meas_etch_sg', 'meas_etch_temp', 'meas_etch_additive', 'meas_etch_amount', 
        'meas_softetch_cu', 'meas_softetch_h2so4', 'meas_softetch_sps', 
        'meas_strip_conc', 'meas_rinse_ph', 'meas_dev_ph', 'meas_dev_conc'
    ]
    df.columns = new_cols

    # 결측치 처리 (재작업사유는 0으로, 수치형은 중앙값으로)
    df['rework_history'] = df['rework_history'].fillna('0')
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 범주형 컬럼 원핫인코딩
    ob_cols = df.select_dtypes('object').columns
    df = pd.get_dummies(df, columns=ob_cols)

    # 타겟 변수 (양산 DES 속도) 분리
    y = df['mass_des_speed']
    x = df.drop(columns=['mass_des_speed'])
    
    # 1. Train_Val 과 Test 로 분리 (80% / 20%)
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # 2. Train_Val 에서 다시 Train 과 Validation 분리 (75% of 80% = 60%, 25% of 80% = 20%)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42)

    # 3. 모든 피처에 대한 스케일링 
    scaler = StandardScaler()
    # Train 데이터 기준으로 스케일러 학습 및 변환
    x_train_scaled = scaler.fit_transform(x_train)
    # Validation과 Test 데이터는 Train에서 학습된 스케일러로 변환만 수행
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    
    # 4. 주성분 분석(PCA)을 통한 차원 축소
    print(f"\n--- PCA 차원 축소 시작 (원본 특성 개수: {x_train.shape[1]}) ---")
    pca = PCA(n_components=n_components, random_state=42)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_val_pca = pca.transform(x_val_scaled)
    x_test_pca = pca.transform(x_test_scaled)
    print(f"-> PCA 적용 후 특성 개수: {x_train_pca.shape[1]} (분산 설명력: {sum(pca.explained_variance_ratio_):.4f})")

    # 결과 데이터프레임 생성
    pca_cols = [f'PC{i+1}' for i in range(x_train_pca.shape[1])]
    
    df_train = pd.DataFrame(x_train_pca, columns=pca_cols, index=x_train.index)
    df_train['mass_des_speed'] = y_train
    
    df_val = pd.DataFrame(x_val_pca, columns=pca_cols, index=x_val.index)
    df_val['mass_des_speed'] = y_val
    
    df_test = pd.DataFrame(x_test_pca, columns=pca_cols, index=x_test.index)
    df_test['mass_des_speed'] = y_test
    
    return df_train, df_val, df_test
