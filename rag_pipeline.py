from langchain_community.document_loaders import JSONLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub

def setup_rag_pipeline(json_path='./rag_data_all.json', docx_path='./rag_opls.docx', db_dir='./chroma_huggingface'):
    """
    주어진 JSON 문서와 DOCX 문서를 모두 읽어 분할하고 임베딩하여 Chroma DB에 저장한 후,
    HuggingFace LLM을 통해 RAG 파이프라인을 세팅합니다.
    """
    print("1. 문서 로드 및 분할 중...")
    
    document_list = []
    
    # 1. JSON 문서 로드
    try:
        json_loader = JSONLoader(
            file_path=json_path,
            jq_schema='.[].content', # 임시 스키마. 필요 시 JSON 구조에 맞게 변경
            text_content=True
        )
        json_docs = json_loader.load()
        document_list.extend(json_docs)
        print(f"-> JSON 파일에서 {len(json_docs)}개의 문서 로드 성공.")
    except Exception as e:
         print(f"JSON 로드 실패. 에러: {e}")

    # 2. DOCX 문서 로드 (rag_opls.docx)
    try:
        # Docx2txtLoader 사용
        docx_loader = Docx2txtLoader(file_path=docx_path)
        docx_docs = docx_loader.load()
        document_list.extend(docx_docs)
        print(f"-> DOCX 파일에서 {len(docx_docs)}개의 문서 로드 성공.")
    except Exception as e:
         print(f"DOCX 로드 실패. 파일이 있는지 확인해주세요. 에러: {e}")

    if not document_list:
        print("로드된 문서가 없습니다. 파이프라인 구성을 중단합니다.")
        return None, None
        
    # 문서 분할 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    document_list = text_splitter.split_documents(document_list)
    print(f"-> 총 {len(document_list)}개의 청크(Chunk)로 분할되었습니다.")

    print("2. 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large-instruct')

    print("3. Chroma 데이터베이스 설정 중...")
    collection_name = 'chroma_rag_data'
    database = Chroma.from_documents(
        documents=document_list,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=db_dir
    )

    print("4. LLM 로컬 로드(4bit 양자화) 중...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    chat_model = HuggingFacePipeline.from_model_id(
        model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        task='text-generation',
        pipeline_kwargs=dict(
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.03
        ),
        model_kwargs={'quantization_config': quantization_config}
    )

    llm = ChatHuggingFace(llm=chat_model)

    # Retrieval QA 템플릿 사용
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    retriever = database.as_retriever(search_kwargs={"k": 1})

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain, llm

def query_rag(retrieval_chain, query):
    """
    구성된 RAG 체인에 질문을 던지고 답변을 반환합니다.
    """
    if retrieval_chain is None:
        print("RAG 체인이 구성되지 않았습니다.")
        return None

    print(f"\n[질문]: {query}")
    ai_message = retrieval_chain.invoke({"input": query})
    print("[답변]:\n", ai_message['answer'])
    return ai_message
