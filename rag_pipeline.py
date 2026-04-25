from langchain_community.document_loaders import JSONLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# 최신 랭체인 표현 언어 모듈
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def setup_rag_pipeline(json_path='./rag_data_all.json', docx_path='./rag_opls.docx', db_dir='./chroma_huggingface'):
    print("1. 문서 로드 및 분할 중...")
    document_list = []
    
    try:
        json_loader = JSONLoader(file_path=json_path, jq_schema='.[].content', text_content=True)
        document_list.extend(json_loader.load())
    except Exception as e:
         print(f"JSON 문서 로드 실패: {e}")

    try:
        docx_loader = Docx2txtLoader(file_path=docx_path)
        document_list.extend(docx_loader.load())
    except Exception as e:
         print(f"DOCX 문서 로드 실패: {e}")

    if not document_list:
        print("로드된 문서가 없습니다.")
        return None, None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    document_list = text_splitter.split_documents(document_list)

    print("2. 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct')

    print("3. 데이터베이스 설정 중...")
    database = Chroma.from_documents(
        documents=document_list,
        embedding=embeddings,
        collection_name='chroma_rag_data',
        persist_directory=db_dir
    )

    print("4. 언어 모델 로드 중...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    chat_model = HuggingFacePipeline.from_model_id(
        model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        task='text-generation',
        pipeline_kwargs=dict(max_new_tokens=1024, do_sample=False, repetition_penalty=1.03),
        model_kwargs={'quantization_config': quantization_config}
    )
    llm = ChatHuggingFace(llm=chat_model)

    # 검색기 설정
    retriever = database.as_retriever(search_kwargs={"k": 1})

    # 지시문 설정
    template = """다음 문맥을 참고해서 질문에 대답해줘.
    
    문맥: {context}
    
    질문: {question}
    답변:"""
    prompt = PromptTemplate.from_template(template)

    # 문서 형태 변환 함수
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 체인 조립
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, llm

def query_rag(rag_chain, query):
    if rag_chain is None:
        print("체인이 구성되지 않았습니다.")
        return None

    print(f"\n[질문]: {query}")
    answer = rag_chain.invoke(query)
    print("[답변]:\n", answer)
    return answer
