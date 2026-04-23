from data_preprocessing import load_and_preprocess_data
from model_training import train_and_evaluate
# from rag_pipeline import setup_rag_pipeline, query_rag

def run_ml_pipeline():
    print("========================================")
    print(" 1. 데이터 파이프라인 (Data Pipeline)   ")
    print("========================================")
    # data_preprocessing 이 이제 분할된 DataFrame 3개를 반환합니다.
    df_train, df_val, df_test = load_and_preprocess_data('./Train_0319.csv')
    print(f"데이터 분리 완료 (Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape})")
    
    print("\n========================================")
    print(" 2. 학습 파이프라인 (ML Pipeline)       ")
    print("========================================")
    # 두 개의 모델(속도 추천 모델, AOI 불량률 예측 모델)을 순차적으로 튜닝하고 평가합니다.
    model_1, model_2 = train_and_evaluate(df_train, df_val, df_test)
    print("ML 파이프라인 처리가 완료되었습니다.")

def run_rag_pipeline():
    print("\n========================================")
    print(" 3. RAG 파이프라인 (LLM Pipeline)       ")
    print("========================================")
    # LLM 및 RAG 구성에 시간이 걸릴 수 있습니다.
    retrieval_chain, llm = setup_rag_pipeline('./tax_with_markdown.docx')
    
    # RAG 질의 테스트
    query = "연봉 5천만원인 거주자의 소득세는?"
    query_rag(retrieval_chain, query)

    # 일반 LLM 질의 테스트 (검색 없이 단순 질의)
    print("\n[일반 LLM 테스트 질의]: 인프런에는 어떤 강의가 있나요?")
    test_ai_message = llm.invoke('인프런에는 어떤 강의가 있나요?')
    print("[일반 LLM 답변]:\n", test_ai_message.content)

if __name__ == "__main__":
    # 머신러닝 파이프라인 실행
    run_ml_pipeline()
    
    # ========================================================
    # RAG(LLM) 파이프라인은 리소스 사용량이 크므로,
    # 필요하실 때만 아래 주석을 해제하여 실행하시기 바랍니다.
    # ========================================================
    # run_rag_pipeline()
