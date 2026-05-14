from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class TemplateInfo(BaseModel):
    match_category: str
    match_cypher: str
    return_pattern_id: str
    return_cypher: str


class Nl2CypherSample(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    qid: str | int
    graph: str
    gold_cypher: str
    gold_match_cypher: Optional[str] = None

    # accept both "question" and "nl_question"
    nl_question: Optional[str] = Field(default=None, validation_alias="question")
    nl_question_raw: Optional[str] = None

    answer_json: Optional[Any] = None
    pred_cypher: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)

    from_template: Optional[TemplateInfo] = None
    categories: Optional[str] = None

    # fields from test.json format
    schema: Optional[str] = None
    data_source: Optional[str] = None
    instance_id: Optional[str] = None
    database_reference_alias: Optional[str] = None
