from data_preprocessing import load_and_preprocess_data
from model_training import train_and_evaluate
from rag_pipeline import setup_rag_pipeline, query_rag

def run_ml_pipeline():
    print("========================================")
    print(" 1. 데이터 파이프라인 (Data Pipeline)   ")
    print("========================================")
    df_train, df_val, df_test = load_and_preprocess_data('./Train_0319.csv')
    print(f"데이터 분리 및 PCA 적용 완료 (Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape})")
    
    print("\n========================================")
    print(" 2. 학습 파이프라인 (ML Pipeline)       ")
    print("========================================")
    # 양산 DES 속도를 예측/추천하는 회귀 모델을 튜닝하고 평가합니다.
    model_1, _ = train_and_evaluate(df_train, df_val, df_test)
    print("ML 파이프라인 처리가 완료되었습니다.")

def run_rag_pipeline():
    from rag_pipeline import setup_rag_pipeline, query_rag
    
    print("\n========================================")
    print(" 3. RAG 파이프라인 (LLM Pipeline)       ")
    print("========================================")
    retrieval_chain, llm = setup_rag_pipeline(json_path='./rag_data_all.json', docx_path='./rag_opls.docx')
    
    # RAG 질의 테스트
    query = "이 문서의 주요 내용은 무엇인가요?"
    query_rag(retrieval_chain, query)

    # 일반 LLM 질의 테스트 (검색 없이 단순 질의)
    print("\n[일반 LLM 테스트 질의]: 요즘 가장 인기 있는 반도체 회사는 어디인가요?")
    test_ai_message = llm.invoke('요즘 가장 인기 있는 반도체 회사는 어디인가요?')
    print("[일반 LLM 답변]:\n", test_ai_message.content)

if __name__ == "__main__":
    # 머신러닝 파이프라인 실행
    run_ml_pipeline()
    
    # ========================================================
    # RAG(LLM) 파이프라인은 리소스 사용량이 크므로,
    # 필요하실 때만 아래 주석을 해제하여 실행하시기 바랍니다.
    # ========================================================
    run_rag_pipeline()
