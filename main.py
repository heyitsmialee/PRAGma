from data_preprocessing import load_and_preprocess_data
from model_training import train_and_evaluate
from rag_pipeline import setup_rag_pipeline, query_rag

def run_ml_pipeline():
    print("========================================")
    print(" 1. 데이터 파이프라인 (Data Pipeline)   ")
    print("========================================")
    x_train, x_test, y_train, y_test = load_and_preprocess_data('./Train_0319.csv')
    print(f"데이터 분리 완료 (Train: {x_train.shape}, Test: {x_test.shape})")
    
    print("\n========================================")
    print(" 2. 학습 파이프라인 (ML Pipeline)       ")
    print("========================================")
    best_model = train_and_evaluate(x_train, x_test, y_train, y_test)
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
    
    # RAG(LLM) 파이프라인 실행 (주석처리)
    # run_rag_pipeline()
