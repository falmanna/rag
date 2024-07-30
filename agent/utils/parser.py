import json

from langchain.output_parsers import OutputFixingParser, RetryOutputParser
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel

from agent.utils.llm import get_llm

CUSTOM_JSON_FIX_PROMPT = """JSON schema instructions:
--------------
{instructions}
--------------
Completion:
--------------
{completion}
--------------

Above, the Completion did not satisfy the constraints given in the JSON schema instructions.
Error:
--------------
{error}
--------------

You must fix return the json output as per the constraints laid out in the JSON schema instructions and nothing more:"""

CUSTOM_JSON_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema instructions below.
```
{schema}
```"""


def _remove_a_key(d, remove_key) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)


def openai_schema(pydantic_object: BaseModel):
    """
    Return the schema in the format of OpenAI's schema as jsonschema

    Note:
        Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

    Returns:
        model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
    """
    schema = pydantic_object.schema()
    parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}
    parameters["required"] = sorted(parameters["properties"])

    if "description" not in schema:
        schema["description"] = (
            f"Correctly extracted `{pydantic_object.__name__}` with all the required parameters with correct types"
        )

    return {
        "name": schema["title"],
        "description": schema["description"],
        "parameters": parameters,
    }


def simplify_schema(pydantic_object: BaseModel):
    def _simplify_schema_dict(schema: dict):
        if schema["type"] == "object" and schema.get("properties"):
            return {
                k: _simplify_schema_dict(v) for k, v in schema["properties"].items()
            }
        elif schema["type"] == "array":
            if schema["items"]["type"] == "object":
                return [_simplify_schema_dict(schema["items"])]
            else:
                return [schema.get("description", "")]
        else:
            return schema.get("description", "")

    schema = pydantic_object.schema()
    return _simplify_schema_dict(schema)


class CustomPydanticOutputParser(PydanticOutputParser):
    def get_format_instructions(self) -> str:
        # # Copy schema to avoid altering original Pydantic schema.
        # schema = {k: v for k, v in self._get_schema(self.pydantic_object).items()}

        # # Remove extraneous fields.
        # reduced_schema = schema
        # if "title" in reduced_schema:
        #     del reduced_schema["title"]
        # if "type" in reduced_schema:
        #     del reduced_schema["type"]
        # # Ensure json in context is well-formed with double quotes.
        # schema_str = json.dumps(reduced_schema)

        # schema = simplify_schema(self.pydantic_object)

        schema = openai_schema(self.pydantic_object)

        schema_str = json.dumps(schema)
        return CUSTOM_JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)


def get_json_parser(
    pydantic_object: BaseModel = None,
    model: str = None,
) -> JsonOutputParser:
    parser = JsonOutputParser(pydantic_object=pydantic_object)

    if model:
        parser = RetryOutputParser.from_llm(
            parser=parser, llm=get_llm(model), max_retries=2
        )

        # parser = OutputFixingParser.from_llm(
        #     parser=parser,
        #     llm=get_llm(model),
        #     prompt=PromptTemplate.from_template(CUSTOM_JSON_FIX_PROMPT),
        # )

    return parser


def get_pydantic_parser(
    pydantic_object: BaseModel = None,
    model: str = None,
) -> PydanticOutputParser:
    parser = PydanticOutputParser(pydantic_object=pydantic_object)

    if model:
        # parser = RetryOutputParser.from_llm(
        #     parser=parser, llm=get_llm(model), max_retries=2
        # )

        parser = OutputFixingParser.from_llm(
            parser=parser,
            llm=get_llm(model),
            prompt=PromptTemplate.from_template(CUSTOM_JSON_FIX_PROMPT),
        )

    return parser


def get_str_parser():
    return StrOutputParser()
