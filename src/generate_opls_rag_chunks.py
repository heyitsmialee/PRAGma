import json
from pathlib import Path


def build_rag_chunks_for_opls_item(item):
    opls_id = item.get("opls_id", "")
    title = item.get("title", "")

    mapped_columns = [
        var.get("standard_column")
        for var in item.get("mapped_variables", [])
        if var.get("standard_column") is not None
    ]

    rag_chunks = []

    # 1. NG 현상 chunk
    if item.get("ng_symptoms"):
        rag_chunks.append({
            "chunk_id": f"{opls_id}_chunk_symptom",
            "chunk_type": "symptom",
            "standard_columns": mapped_columns,
            "content": (
                f"{title} 항목의 주요 NG 현상은 "
                + ", ".join(item["ng_symptoms"])
                + "이다."
            )
        })

    # 2. 원인 chunk
    if item.get("possible_causes"):
        rag_chunks.append({
            "chunk_id": f"{opls_id}_chunk_cause",
            "chunk_type": "cause",
            "standard_columns": mapped_columns,
            "content": (
                f"{title} 항목의 예상 원인은 "
                + ", ".join(item["possible_causes"])
                + "이다."
            )
        })

    # 3. 조치 chunk
    if item.get("inspection_actions"):
        rag_chunks.append({
            "chunk_id": f"{opls_id}_chunk_action",
            "chunk_type": "action",
            "standard_columns": mapped_columns,
            "content": (
                f"{title} 발생 시 점검 및 조치 사항은 "
                + ", ".join(item["inspection_actions"])
                + "이다."
            )
        })

    # 4. Rule chunk
    for idx, rule in enumerate(item.get("rules", []), start=1):
        condition = rule.get("condition", {})
        effect = rule.get("effect", {})

        cond_col = condition.get("standard_column", "")
        cond_change = condition.get("change", "")
        eff_col = effect.get("standard_column", effect.get("paper_term", ""))
        eff_change = effect.get("change", "")

        content = (
            f"{title} 기준으로 {cond_col} 값이 {cond_change}하면 "
            f"{eff_col}이 {eff_change}할 수 있다."
        )

        if rule.get("korean_rule_explanation"):
            content += f" {rule['korean_rule_explanation']}"

        if rule.get("risk"):
            content += f" 관련 리스크는 {rule['risk']}이다."

        rag_chunks.append({
            "chunk_id": f"{opls_id}_chunk_rule_{idx:02d}",
            "chunk_type": "rule",
            "standard_columns": [
                col for col in [cond_col, eff_col] if col
            ],
            "content": content
        })

    # 5. 핵심 포인트 chunk
    if item.get("core_point"):
        rag_chunks.append({
            "chunk_id": f"{opls_id}_chunk_core_point",
            "chunk_type": "core_point",
            "standard_columns": mapped_columns,
            "content": f"{title}의 핵심 포인트는 {item['core_point']}"
        })

    return rag_chunks


def add_rag_chunks_to_opls_json(
    input_path="data/opls_process_knowledge.json",
    output_path="data/opls_process_knowledge.json"
):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        item["rag_chunks"] = build_rag_chunks_for_opls_item(item)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"rag_chunks 추가 완료: {output_path}")


if __name__ == "__main__":
    add_rag_chunks_to_opls_json(
        input_path="data/opls_process_knowledge.json",
        output_path="data/opls_process_knowledge.json"
    )
