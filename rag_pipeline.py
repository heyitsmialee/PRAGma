import torch

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub


def setup_rag_pipeline(
    docx_path="./tax_with_markdown.docx",
    db_dir="./chroma_huggingface"
):
    """
    주어진 문서를 읽어 분할하고 임베딩하여 Chroma DB에 저장한 후,
    EXAONE 3.5 기반 HuggingFace LLM으로 RAG 파이프라인을 세팅합니다.
    """

    print("1. 문서 로드 및 분할 중...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    loader = Docx2txtLoader(docx_path)
    document_list = loader.load_and_split(text_splitter)

    print("2. 임베딩 모델 로드 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct"
    )

    print("3. Chroma 데이터베이스 설정 중...")
    collection_name = "chroma_tax"

    database = Chroma.from_documents(
        documents=document_list,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=db_dir
    )

    print("4. LLM 로컬 로드 중: EXAONE 3.5 7.8B Instruct + 4bit 양자화")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    chat_model = HuggingFacePipeline.from_model_id(
        model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        task="text-generation",
        pipeline_kwargs=dict(
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.03,
            return_full_text=False
        ),
        model_kwargs={
            "quantization_config": quantization_config,
            "trust_remote_code": True
        }
    )

    llm = ChatHuggingFace(llm=chat_model)

    print("5. Retriever 및 RAG Chain 구성 중...")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    retriever = database.as_retriever(
        search_kwargs={"k": 3}
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm,
        retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever,
        combine_docs_chain
    )

    print("RAG 파이프라인 세팅 완료")

    return retrieval_chain, llm


def query_rag(retrieval_chain, query):
    """
    구성된 RAG 체인에 질문을 던지고 답변을 반환합니다.
    """

    print(f"\n[질문]: {query}")

    ai_message = retrieval_chain.invoke({
        "input": query
    })

    print("[답변]:\n", ai_message["answer"])

    return ai_message
