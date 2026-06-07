
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
NOTEBOOK_DIR = BASE_DIR / "notebooks"
MODEL_DIR = BASE_DIR / "models"

MODEL_CANDIDATES = [
    NOTEBOOK_DIR / "best_LightGBM_mass_speed_regressor.pkl",
    MODEL_DIR / "best_LightGBM_mass_speed_regressor.pkl",
    BASE_DIR / "best_LightGBM_mass_speed_regressor.pkl",
]

SHAP_CANDIDATES = [
    NOTEBOOK_DIR / "shap_analysis_for_rag.md",
    DATA_DIR / "shap_analysis_for_rag.md",
    BASE_DIR / "shap_analysis_for_rag.md",
]

CSV_PATH = DATA_DIR / "Train_0319.csv"
PAPER_JSON_PATH = DATA_DIR / "rag_data_all.json"
OPLS_JSON_PATH = DATA_DIR / "opls_process_knowledge.json"


st.set_page_config(page_title="PRAGma", page_icon="⚙️", layout="wide")

st.markdown("""
<style>
.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}

h1 {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
}

.kpi-card {
    padding: 18px 20px;
    border-radius: 18px;
    color: white;
    min-height: 145px;
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12);
}

.kpi-green {
    background: linear-gradient(135deg, #28544d 0%, #1e7565 100%);
}

.kpi-orange {
    background: linear-gradient(135deg, #654321 0%, #b26a12 100%);
}

.kpi-red {
    background: linear-gradient(135deg, #5f2d2d 0%, #b83434 100%);
}

.kpi-title {
    font-size: 15px;
    font-weight: 700;
    opacity: 0.85;
    margin-bottom: 14px;
}

.kpi-value {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 12px;
}

.kpi-status {
    display: inline-block;
    background: rgba(255,255,255,0.22);
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
}

.left-menu {
    position: sticky;
    top: 1rem;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px;
}

.menu-title {
    font-weight: 800;
    margin-bottom: 12px;
}

.menu-item {
    padding: 8px 0;
    color: #334155;
    font-size: 14px;
}

.alert-red {
    background:#fff5f5;
    border-left:6px solid #ef4444;
    padding:16px;
    border-radius:12px;
    margin-bottom:14px;
}

.alert-yellow {
    background:#fffaf0;
    border-left:6px solid #f59e0b;
    padding:16px;
    border-radius:12px;
    margin-bottom:14px;
}

.result-box {
    background:#f8fafc;
    border:1px solid #e5e7eb;
    padding:16px;
    border-radius:12px;
    margin-bottom:14px;
}
</style>
""", unsafe_allow_html=True)


def find_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


@st.cache_resource
def load_resources():
    model_path = find_path(MODEL_CANDIDATES)
    if model_path is None:
        raise FileNotFoundError("best_LightGBM_mass_speed_regressor.pkl 파일을 찾을 수 없습니다.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(CSV_PATH, encoding="cp949")

    paper_rules = []
    if PAPER_JSON_PATH.exists():
        with open(PAPER_JSON_PATH, "r", encoding="utf-8") as f:
            paper_rules = json.load(f)

    opls_rules = []
    if OPLS_JSON_PATH.exists():
        with open(OPLS_JSON_PATH, "r", encoding="utf-8") as f:
            opls_rules = json.load(f)

    shap_path = find_path(SHAP_CANDIDATES)
    shap_text = shap_path.read_text(encoding="utf-8") if shap_path else ""

    return model, df, paper_rules, opls_rules, shap_text, model_path


model, df_mes, paper_rules, opls_rules, shap_text, model_path = load_resources()

try:
    MODEL_FEATURES = model.feature_name_
except Exception:
    MODEL_FEATURES = model.booster_.feature_name()


CONTINUOUS_MAP = {
    "Cu 표면두께 Max_Val": "cu_thick_max",
    "Cu 표면두께 AVG_VAL": "cu_thick_avg",
    "Cu 표면두께 Min_Val": "cu_thick_min",
    "Cu 표면두께 Std_Val": "cu_thick_std",
    "Cu 표면두께 Median_Val": "cu_thick_median",
}

SUFFIX_MAP = {
    "Etch factor": "etch_factor",
    "Etching(염화동) - Cu": "meas_etch_cu",
    "Etching(염화동) - HCl": "meas_etch_hcl",
    "Etching(염화동) - 비중": "meas_etch_sg",
    "Etching(염화동) - 온도": "meas_etch_temp",
    "Etching-첨가제(HB-120EF)": "meas_etch_additive",
    "Etching량": "meas_etch_amount",
    "Soft Etch - Cu": "meas_softetch_cu",
    "Soft Etch - H2SO4": "meas_softetch_h2so4",
    "Soft Etch - SPS": "meas_softetch_sps",
    "박리액 - 농도": "meas_strip_conc",
    "수세수 - pH": "meas_rinse_ph",
    "현상액 - pH": "meas_dev_ph",
    "현상액 - 농도": "meas_dev_conc",
}

for col in df_mes.columns:
    if "분석치" in col or "分" in col:
        suffix = col.split("_", 1)[-1] if "_" in col else col
        if suffix in SUFFIX_MAP:
            CONTINUOUS_MAP[col] = SUFFIX_MAP[suffix]


CATEGORICAL_COLS = {
    "재작업사유": {
        "prefix": "rework_history",
        "values": ["Unknown", "기타", "기판 겹침", "두께 미달", "딤플", "설비 에러"],
        "sep": "_",
    },
    "노광 설비정보": {
        "prefix": "expo_eq_id",
        "values": [f"EXP-{i:03d}" for i in range(1, 8)],
        "sep": "_",
    },
    "DES 설비정보": {
        "prefix": "des_eq_id",
        "values": [f"DES-{i:03d}" for i in range(1, 7)],
        "sep": "_",
    },
    "정면 설비정보": {
        "prefix": "brush_eq_id",
        "values": [f"PRE-{i:03d}" for i in range(1, 8)],
        "sep": "_",
    },
}


OPLS_BOUNDS = {
    "meas_etch_temp": {"name": "에칭 온도", "lcl": 44.5, "sl": 48.0, "ucl": 53.0, "unit": "°C"},
    "meas_etch_sg": {"name": "에칭 비중", "lcl": 1.32, "sl": 1.37, "ucl": 1.42, "unit": ""},
    "meas_etch_cu": {"name": "에칭 Cu 농도", "lcl": 125.0, "sl": 155.0, "ucl": 185.0, "unit": "g/L"},
    "meas_etch_hcl": {"name": "에칭 HCl", "lcl": 0.3, "sl": 0.5, "ucl": 0.7, "unit": "N"},
    "meas_etch_additive": {"name": "에칭 첨가제", "lcl": 2.6, "sl": 3.0, "ucl": 3.4, "unit": "g/L"},
}


def get_lot_data(lot_no):
    rows = df_mes[df_mes["LOT"] == lot_no]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def prepare_features(lot_data):
    row = {}

    for csv_col, feat in CONTINUOUS_MAP.items():
        val = lot_data.get(csv_col, np.nan)
        row[feat] = float(val) if val is not None and str(val) not in ["", "nan", "None"] else np.nan

    for csv_col, cfg in CATEGORICAL_COLS.items():
        raw = str(lot_data.get(csv_col, "")).strip()
        norm = raw.replace(" ", "_") if cfg["prefix"] == "rework_history" else raw

        for value in cfg["values"]:
            norm_value = value.replace(" ", "_") if cfg["prefix"] == "rework_history" else value
            key = f"{cfg['prefix']}{cfg['sep']}{norm_value}"
            row[key] = 1 if norm == norm_value else 0

    for feat in MODEL_FEATURES:
        if feat not in row:
            row[feat] = 0

    return pd.DataFrame([row])[MODEL_FEATURES]


def predict_speed(lot_data):
    X = prepare_features(lot_data)
    pred = float(model.predict(X)[0])
    return pred, X


def check_opls(X):
    result = []

    for feat, bound in OPLS_BOUNDS.items():
        if feat not in X.columns:
            continue

        value = X[feat].iloc[0]
        if pd.isna(value):
            continue

        status = "정상"
        level = "ok"

        if value > bound["ucl"]:
            status = "UCL 초과"
            level = "high"
        elif value < bound["lcl"]:
            status = "LCL 미달"
            level = "high"
        elif value > bound["sl"] * 1.02:
            status = "SL 상향 근접"
            level = "mid"
        elif value < bound["sl"] * 0.98:
            status = "SL 하향 근접"
            level = "mid"

        result.append({
            "피처": feat,
            "항목": bound["name"],
            "현재값": round(float(value), 4),
            "LCL": bound["lcl"],
            "SL": bound["sl"],
            "UCL": bound["ucl"],
            "단위": bound["unit"],
            "상태": status,
            "위험도": level,
        })

    return result


def process_summary(alerts):
    high = [a for a in alerts if a["위험도"] == "high"]
    mid = [a for a in alerts if a["위험도"] == "mid"]

    if high:
        return "위험", f"{high[0]['항목']} {high[0]['상태']}", "즉시 점검이 필요합니다."
    if mid:
        return "주의", f"{mid[0]['항목']} {mid[0]['상태']}", "추세 확인 및 사전 점검이 필요합니다."
    return "정상", "OPLS 이탈 없음", "주요 관리 기준 내에 있습니다."


def recommend_speed(pred, actual):
    if not isinstance(actual, (int, float, np.integer, np.floating)):
        return "확인 불가", None, "실제 속도 데이터가 없어 예측값만 참고합니다."

    diff = pred - actual

    if abs(diff) < 0.03:
        return "유지", diff, "현재 속도와 모델 권장 속도 차이가 작아 유지 권장입니다."
    if diff > 0:
        return "상향", diff, f"현재 대비 약 {diff:.3f} m/min 상향 검토 가능합니다."
    return "하향", diff, f"현재 대비 약 {abs(diff):.3f} m/min 하향 검토가 필요합니다."


def search_knowledge(question, top_k=4):
    q = question.lower()
    docs = []

    for item in paper_rules:
        docs.append(("논문 Rule", json.dumps(item, ensure_ascii=False)))

    for item in opls_rules:
        docs.append(("OPLS Rule", json.dumps(item, ensure_ascii=False)))
        for chunk in item.get("rag_chunks", []):
            docs.append(("OPLS Chunk", chunk.get("content", "")))

    if shap_text:
        docs.append(("SHAP", shap_text[:3000]))

    keys = ["온도", "비중", "cu", "구리", "hcl", "첨가제", "선폭", "과에칭", "미에칭", "속도", "etch", "factor", "ph", "박리"]
    scored = []

    for doc_type, text in docs:
        low = text.lower()
        score = 0

        for key in keys:
            if key in q and key in low:
                score += 2

        for word in q.replace(",", " ").replace(".", " ").split():
            if len(word) >= 2 and word in low:
                score += 1

        if score > 0:
            scored.append((score, doc_type, text))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]


def chatbot_answer(question):
    docs = search_knowledge(question)

    q = question.lower()

    action_map = {
        "cu": [
            "Auto Drain 동작 여부를 확인합니다.",
            "신액 보충 상태와 배액 라인 막힘 여부를 점검합니다.",
            "Cu 농도 상승이 Etch factor 저하로 이어지는지 함께 확인합니다.",
        ],
        "구리": [
            "Auto Drain 동작 여부를 확인합니다.",
            "신액 보충 상태와 배액 라인 막힘 여부를 점검합니다.",
            "Cu 농도 상승이 Etch factor 저하로 이어지는지 함께 확인합니다.",
        ],
        "온도": [
            "칠러 및 히터 제어 상태를 확인합니다.",
            "에칭액 온도 센서값과 실제 측정값을 교차 확인합니다.",
            "온도 상승 시 과에칭 가능성이 있으므로 컨베이어 속도 보정 여부를 검토합니다.",
        ],
        "비중": [
            "비중계 센서 오염 또는 슬러지 부착 여부를 확인합니다.",
            "약액 농도 보정 및 보충 이력을 확인합니다.",
            "비중 상승 시 에칭 반응 속도 변화 가능성을 함께 점검합니다.",
        ],
        "첨가제": [
            "첨가제 토출량과 펌프 동작 상태를 확인합니다.",
            "펌프 에어록 또는 노즐 막힘 여부를 점검합니다.",
            "첨가제 농도 저하 시 미에칭 또는 선폭 불균일 가능성을 확인합니다.",
        ],
        "etch factor": [
            "Cu 농도, 첨가제 농도, 스프레이 상태를 함께 확인합니다.",
            "상하부 노즐 분사 균일성을 점검합니다.",
            "Etch factor 저하 시 선폭 품질 변동 가능성을 확인합니다.",
        ],
        "선폭": [
            "Cu 표면두께 산포와 Etch factor를 우선 확인합니다.",
            "노광/현상 조건과 DES 속도 편차를 함께 점검합니다.",
            "선폭 OFFSET 방향에 따라 과에칭 또는 미에칭 가능성을 구분합니다.",
        ],
        "ph": [
            "현상액 pH 센서값과 실측값을 비교합니다.",
            "K2CO3 보충 라인과 탱크 수위를 확인합니다.",
            "pH 하락이 현상 불량 또는 잔사 발생으로 이어지는지 확인합니다.",
        ],
        "박리": [
            "박리액 농도와 순환 펌프 상태를 확인합니다.",
            "필터 차압 및 스퀴지 롤러 상태를 점검합니다.",
            "박리 잔사 발생 시 수세수 pH와 세정 조건을 함께 확인합니다.",
        ],
    }

    selected_actions = None

    for key, actions in action_map.items():
        if key in q:
            selected_actions = actions
            break

    if selected_actions is None:
        selected_actions = [
            "질문과 관련된 공정 변수의 OPLS 이탈 여부를 먼저 확인합니다.",
            "센서값과 실측값을 교차 확인합니다.",
            "약품 농도, 설비 상태, 노즐/펌프/배액 라인을 순차적으로 점검합니다.",
        ]

    lines = []
    lines.append("### 조치 방향")

    for action in selected_actions:
        lines.append(f"- {action}")

    lines.append("")
    lines.append("### 관련 근거")

    if not docs:
        lines.append("- 관련 문서를 찾지 못했습니다.")
    else:
        for idx, (_, doc_type, text) in enumerate(docs[:3], start=1):
            text = text.replace("\n", " ")

            if len(text) > 350:
                text = text[:350] + "..."

            lines.append(f"{idx}. **{doc_type}**: {text}")

    return "\n".join(lines)

def rag_answer(question, alerts=None, pred=None):
    return chatbot_answer(question)


def trouble_actions(trouble):
    table = {
        "과에칭": ["에칭 온도/비중 UCL 초과 여부 확인", "Cu/HCl 농도 확인", "컨베이어 속도 하향 여부 확인", "선폭 OFFSET 마이너스 변동 확인"],
        "미에칭": ["Etch factor 저하 확인", "Cu 농도 과다 및 배액 불량 확인", "첨가제 농도 확인", "노즐 막힘 및 스프레이 압력 확인"],
        "선폭 불균일": ["Cu 표면두께 산포 확인", "Soft Etch SPS/H2SO4 확인", "롤러 마모 및 이송 속도 편차 확인"],
        "Cu 농도 과다": ["Auto Drain 확인", "신액 보충 상태 확인", "배액 라인 막힘 확인", "Etch factor 저하 동시 확인"],
        "Etch factor 저하": ["첨가제 실제 토출량 확인", "펌프 에어록 제거", "상하부 스프레이 압력 비교", "노즐 세척"],
        "현상액 pH 이탈": ["현상액 농도/pH 트렌드 확인", "K2CO3 보충 라인 확인", "보충 탱크 수위 확인"],
        "박리 잔사": ["박리액 농도 확인", "순환 펌프 필터 차압 확인", "스퀴지 롤러 상태 확인", "수세수 pH 확인"],
    }

    return table.get(trouble, [])


@st.cache_data
def make_monitoring_data():
    n = 30
    x = np.arange(n)

    temp = 48.0 + 0.10 * np.sin(x / 2) + np.linspace(-0.08, 0.15, n)
    sg = 1.370 + 0.004 * np.sin(x / 3) + np.linspace(0.000, 0.012, n)
    cu = 176 + 0.8 * x + 2.2 * np.sin(x / 4)
    additive = 2.95 + 0.035 * np.sin(x / 3) - np.linspace(0, 0.04, n)

    return pd.DataFrame({
        "시간": [f"T-{(n-i)*10}s" for i in range(n)],
        "에칭 온도": temp,
        "에칭 비중": sg,
        "에칭 Cu 농도": cu,
        "에칭 첨가제": additive,
    })


def make_deviation_chart(df, col, sl):
    return pd.DataFrame({
        "SL 대비 편차(%)": ((df[col] - sl) / sl) * 100
    })


def render_kpi_card(title, value, unit, status, color):
    st.markdown(
        f"""
        <div class="kpi-card {color}">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}<span style="font-size:18px;"> {unit}</span></div>
            <div class="kpi-status">{status}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_dashboard():
    left_menu, main = st.columns([0.18, 0.82])

    with left_menu:
        st.markdown("""
        <div class="left-menu">
            <div class="menu-title">공정 대시보드</div>
            <div class="menu-item">〽 실시간 모니터링</div>
            <div class="menu-item">🔍 LOT 분석 / 최적 속도</div>
        </div>
        """, unsafe_allow_html=True)

    with main:
        st.subheader("〽 실시간 모니터링")

        df = make_monitoring_data()

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card("에칭 온도", f"{df['에칭 온도'].iloc[-1]:.2f}", "°C", "정상", "kpi-green")
        with k2:
            render_kpi_card("에칭 비중", f"{df['에칭 비중'].iloc[-1]:.3f}", "", "정상", "kpi-green")
        with k3:
            render_kpi_card("에칭 Cu 농도", f"{df['에칭 Cu 농도'].iloc[-1]:.1f}", "g/L", "UCL 초과", "kpi-red")
        with k4:
            render_kpi_card("에칭 첨가제", f"{df['에칭 첨가제'].iloc[-1]:.2f}", "g/L", "주의", "kpi-orange")

        st.write("")

        trend_col, alarm_col = st.columns([1.15, 1])

        with trend_col:
            st.markdown("#### 공정 트렌드")

            c1, c2 = st.columns(2)

            with c1:
                st.caption("에칭 온도 - SL 대비 편차")
                st.line_chart(make_deviation_chart(df, "에칭 온도", 48.0), height=180)

                st.caption("에칭 Cu 농도 - SL 대비 편차")
                st.line_chart(make_deviation_chart(df, "에칭 Cu 농도", 155.0), height=180)

            with c2:
                st.caption("에칭 비중 - SL 대비 편차")
                st.line_chart(make_deviation_chart(df, "에칭 비중", 1.37), height=180)

                st.caption("에칭 첨가제 - SL 대비 편차")
                st.line_chart(make_deviation_chart(df, "에칭 첨가제", 3.0), height=180)

        with alarm_col:
            st.markdown("#### 🔔 공정 알람")

            a1, a2, a3 = st.columns(3)
            a1.metric("위험", "1")
            a2.metric("경고", "2")
            a3.metric("해제", "5")

            st.markdown("""
            <div class="alert-red">
            <b>에칭 Cu 농도 UCL 초과 — DES-003</b><br><br>
            현재값 190.0 g/L | UCL 185.0 초과<br><br>
            권장 조치:
            <ul>
                <li>Auto Drain 확인</li>
                <li>신액 보충 진행</li>
                <li>Etch factor 동시 점검</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="alert-yellow">
            <b>에칭 첨가제 SL 하향 근접</b><br><br>
            첨가제 농도 저하 추세 확인<br><br>
            권장 조치:
            <ul>
                <li>첨가제 토출량 확인</li>
                <li>펌프 에어록 제거</li>
                <li>노즐 상태 점검</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.subheader("🔍 LOT 분석 / 최적 속도 도출")

        lot_left, lot_right = st.columns([1, 1.6])

        with lot_left:
            lot_no = st.text_input("LOT 번호", value="A20000")
            question = st.text_area(
                "분석 질문",
                value="이 LOT의 에칭 공정 상태와 개선 방향을 알려주세요.",
                height=120
            )

            lot_preview = get_lot_data(lot_no)

            if lot_preview is not None:
                with st.expander("LOT 기본 정보", expanded=True):
                    info_col1, info_col2 = st.columns(2)

                    with info_col1:
                        st.text_input("제품군", value=str(lot_preview.get("제품군", "N/A")), disabled=True)
                        st.text_input("거래처", value=str(lot_preview.get("거래처", "N/A")), disabled=True)
                        st.text_input("LAYER", value=str(lot_preview.get("LAYER", "N/A")), disabled=True)

                    with info_col2:
                        st.text_input("공법구분", value=str(lot_preview.get("공법구분", "N/A")), disabled=True)
                        st.text_input("도금구분", value=str(lot_preview.get("도금구분", "N/A")), disabled=True)
                        st.text_input("DRY FILM 정보", value=str(lot_preview.get("DRY FILM 정보", "N/A")), disabled=True)

                with st.expander("설비 정보", expanded=False):
                    st.text_input("노광 설비정보", value=str(lot_preview.get("노광 설비정보", "N/A")), disabled=True)
                    st.text_input("DES 설비정보", value=str(lot_preview.get("DES 설비정보", "N/A")), disabled=True)
                    st.text_input("정면 설비정보", value=str(lot_preview.get("정면 설비정보", "N/A")), disabled=True)

            run = st.button("분석 실행", type="primary", use_container_width=True)
            st.caption(f"사용 모델: {model_path.name}")

        with lot_right:
            if run:
                lot = get_lot_data(lot_no)

                if lot is None:
                    st.error(f"{lot_no} LOT를 찾을 수 없습니다.")
                else:
                    pred, X = predict_speed(lot)
                    actual = lot.get("부식 Speed")
                    alerts = check_opls(X)
                    risk, issue, desc = process_summary(alerts)
                    direction, diff, message = recommend_speed(pred, actual)

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("예측 DES 속도", f"{pred:.4f} m/min")
                    c2.metric("실제 속도", f"{actual} m/min")

                    if isinstance(actual, (int, float, np.integer, np.floating)) and actual != 0:
                        c3.metric("오차율", f"{abs(pred - actual) / actual * 100:.2f}%")
                    else:
                        c3.metric("오차율", "N/A")

                    c4.metric("조정 방향", direction)

                    st.markdown(
                        f"""
                        <div class="result-box">
                        <b>{issue}</b><br>{desc}<br><br>
                        <b>속도 조정 제안</b><br>
                        {message}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.dataframe(pd.DataFrame(alerts), use_container_width=True)

                    st.markdown("#### AI 공정 분석")
                    st.markdown(rag_answer(question, alerts, pred))
            else:
                st.info("LOT 번호를 입력하고 분석 실행을 눌러주세요.")


top_left, top_right = st.columns([2, 1])

with top_left:
    st.title("⚙️ PRAGma")

with top_right:
    st.write("")
    st.success(f"● 실시간  |  {datetime.now().strftime('%H:%M:%S')}")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "🏭 공정 대시보드",
    "🔧 트러블 대응",
    "💬 공정 지식 챗봇",
])


with tab1:
    render_dashboard()


with tab2:
    st.subheader("트러블 대응")

    left, right = st.columns([1, 1.4])

    with left:
        trouble = st.radio(
            "트러블 유형 선택",
            [
                "과에칭",
                "미에칭",
                "선폭 불균일",
                "Cu 농도 과다",
                "Etch factor 저하",
                "현상액 pH 이탈",
                "박리 잔사",
            ]
        )

        st.markdown(f"#### AI 조치 제안 — {trouble}")

        for idx, action in enumerate(trouble_actions(trouble), start=1):
            st.markdown(f"**{idx}. {action}**")

    with right:
        q = st.text_area(
            "공정 지식 검색",
            value=f"{trouble} 발생 시 원인과 조치 방향을 알려줘.",
            height=120
        )

        if st.button("지식 검색", type="primary", use_container_width=True):
            st.markdown(chatbot_answer(q))


with tab3:
    st.subheader("공정 지식 챗봇")

    examples = [
        "Cu 농도가 UCL 초과하면 어떤 조치를 해야 하나요?",
        "Etch factor가 낮을 때 원인은 무엇인가요?",
        "선폭 OFFSET이 마이너스일 때 무엇을 확인해야 하나요?",
        "현상액 pH가 낮으면 어떤 문제가 생기나요?",
    ]

    if "chatbot_question" not in st.session_state:
        st.session_state.chatbot_question = "Etching 온도가 높으면 어떤 문제가 생기나요?"

    cols = st.columns(4)

    for idx, example in enumerate(examples):
        if cols[idx].button(example):
            st.session_state.chatbot_question = example

    q = st.text_input(
        "질문을 입력하세요",
        key="chatbot_question"
    )

    if st.button("답변 생성"):
        st.markdown(chatbot_answer(q))
