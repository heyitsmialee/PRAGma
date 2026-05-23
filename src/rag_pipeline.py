import os
import json
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def load_json_as_documents(json_path: str, knowledge_type: str) -> List[Document]:
    documents = []

    if not os.path.exists(json_path):
        print(f"JSON 파일 없음: {json_path}")
        return documents

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        content = json.dumps(item, ensure_ascii=False, indent=2)

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": json_path,
                    "type": knowledge_type,
                    "id": item.get("paper_id", item.get("opls_id", ""))
                }
            )
        )

    return documents


def load_markdown_as_documents(md_path: str) -> List[Document]:
    if not os.path.exists(md_path):
        print(f"Markdown 파일 없음: {md_path}")
        return []

    loader = TextLoader(file_path=md_path, encoding="utf-8")
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = md_path
        doc.metadata["type"] = "shap_analysis"

    return docs


def setup_rag_pipeline(
    paper_json_path="./rag_data_all.json",
    opls_json_path="./opls_process_knowledge.json",
    shap_md_path="./shap_analysis_for_rag.md",
    db_dir="./chroma_huggingface",
    rebuild_db=True
):
    print("1. 문서 로드 중...")

    document_list = []

    document_list.extend(
        load_json_as_documents(
            paper_json_path,
            knowledge_type="paper_rule"
        )
    )

    document_list.extend(
        load_json_as_documents(
            opls_json_path,
            knowledge_type="opls_process_rule"
        )
    )

    document_list.extend(
        load_markdown_as_documents(shap_md_path)
    )

    if not document_list:
        print("로드된 문서가 없습니다.")
        return None, None

    print(f"로드된 문서 수: {len(document_list)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    split_docs = text_splitter.split_documents(document_list)

    print(f"분할된 Chunk 수: {len(split_docs)}")

    print("2. 임베딩 모델 로드 중...")

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large-instruct"
    )

    print("3. Chroma DB 설정 중...")

    if rebuild_db:
        database = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name="chroma_rag_data",
            persist_directory=db_dir
        )
    else:
        database = Chroma(
            collection_name="chroma_rag_data",
            embedding_function=embeddings,
            persist_directory=db_dir
        )

    print("4. EXAONE 모델 로드 중...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    chat_model = HuggingFacePipeline.from_model_id(
        model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 1024,
            "do_sample": False,
            "repetition_penalty": 1.03
        },
        model_kwargs={
            "quantization_config": quantization_config
        }
    )

    llm = ChatHuggingFace(llm=chat_model)

    retriever = database.as_retriever(
        search_kwargs={"k": 3}
    )

    template = """
다음 문맥을 참고하여 질문에 답변해 주세요.

문맥에는 논문 기반 공정 rule, 현업 OPLS 공정 조치 정보,
SHAP 기반 모델 해석 정보가 포함될 수 있습니다.

답변 시 아래 내용을 중심으로 정리해 주세요.
- 질문에 대한 핵심 답변
- 관련 공정 변수
- 모델 또는 문헌 기반 근거
- 필요 시 공정 조정 방향

문맥:
{context}

질문:
{question}

답변:
"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(
            f"[source={doc.metadata.get('source', '')}, type={doc.metadata.get('type', '')}]\n{doc.page_content}"
            for doc in docs
        )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG Pipeline 구성 완료")
    return rag_chain, llm


def query_rag(rag_chain, query):
    if rag_chain is None:
        print("체인이 구성되지 않았습니다.")
        return None

    print(f"\n[질문]: {query}")

    answer = rag_chain.invoke(query)

    print("[답변]:\n")
    print(answer)

    return answer


if __name__ == "__main__":
    rag_chain, llm = setup_rag_pipeline(
        paper_json_path="./rag_data_all.json",
        opls_json_path="./opls_process_knowledge.json",
        shap_md_path="./shap_analysis_for_rag.md",
        db_dir="./chroma_huggingface",
        rebuild_db=True
    )

    query_rag(
        rag_chain,
        "Etching 온도와 비중이 높을 때 어떤 문제가 발생할 수 있고 어떻게 조치해야 하나요?"
    )
