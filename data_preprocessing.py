import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path='./Train_0319.csv'):
    """
    데이터 로드, 컬럼명 변경, 결측치 처리, 파생변수 생성 및 원핫인코딩을 수행합니다.
    """
    raw_df = pd.read_csv(csv_path, encoding='cp949')
    
    # 필요한 컬럼만 추출
    df = raw_df[[ 
        'Result Ng2', '초물 DES 속도', 'DRY FILM 정보', '선폭 OFFSET', '양산DES 속도', 
        '통합코드', '거래처', '제품군', '공법구분', 'LAYER', '도금구분', '노광 설비정보', 
        'DES 설비정보', '정면 설비정보', 'Cu 표면두께 Max_Val', 'Cu 표면두께 AVG_VAL', 
        'Cu 표면두께 Min_Val', 'Cu 표면두께 Std_Val', 'Cu 표면두께 Median_Val', 
        '재작업사유', '분석치_Etch factor', '분석치_Etching(염화동) - Cu', 
        '분석치_Etching(염화동) - HCl', '분석치_Etching(염화동) - 비중', 
        '분석치_Etching(염화동) - 온도', '분석치_Etching-첨가제(HB-120EF)', 
        '분석치_Etching량', '분석치_Soft Etch - Cu', '분석치_Soft Etch - H2SO4', 
        '분석치_Soft Etch - SPS', '분석치_박리액 - 농도', '분석치_수세수 - pH', 
        '분석치_현상액 - pH', '분석치_현상액 - 농도'
    ]].copy()

    # 영문/단축 컬럼명으로 변경
    new_cols = [
        'defect_code', 'first_des_speed', 'df_type', 'width_offset', 'mass_des_speed', 
        'product_code', 'customer', 'product_family', 'process_type', 'layer_count', 
        'plating_type', 'expo_eq_id', 'des_eq_id', 'brush_eq_id', 'cu_thick_max', 
        'cu_thick_avg', 'cu_thick_min', 'cu_thick_std', 'cu_thick_median', 
        'rework_history', 'etch_factor', 'meas_etch_cu', 'meas_etch_hcl', 
        'meas_etch_sg', 'meas_etch_temp', 'meas_etch_additive', 'meas_etch_amount', 
        'meas_softetch_cu', 'meas_softetch_h2so4', 'meas_softetch_sps', 
        'meas_strip_conc', 'meas_rinse_ph', 'meas_dev_ph', 'meas_dev_conc'
    ]
    df.columns = new_cols

    # 불량 코드 그룹화
    _defect_map = {
        '/OK': '/OK',
        '/기타 불량': '/OK',
        '/패턴폭미달': '/패턴폭미달',
        '/미부식': '/미부식',
        '/미부식/패턴폭미달': '/미부식/패턴폭미달',
        '/미부식/패턴폭미달/수축률': '/미부식/패턴폭미달',
    }
    df['defect_group'] = df['defect_code'].map(_defect_map).fillna(df['defect_code'])
    
    # 파생변수 생성 (target 대비 offset)
    df['speed_offset'] = df['mass_des_speed'] - df['first_des_speed']
    df['rework_history'] = df['rework_history'].fillna('0')

    # 결측치 처리 (수치형은 중앙값)
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 범주형 컬럼 원핫인코딩
    ob_cols = df.select_dtypes('object').columns
    df = pd.get_dummies(df, columns=ob_cols)

    # X, y 분리
    X = df.drop(columns=['mass_des_speed', 'speed_offset'], errors='ignore')
    y = df['speed_offset']

    # Train / Test Split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return x_train, x_test, y_train, y_test
