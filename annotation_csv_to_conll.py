import ast
import json
from typing import TypedDict, cast

import pandas as pd


class Annotation(TypedDict):
    id: str
    entity: str
    start_position: int
    end_position: int
    concept_category: str


class CsvDataEntry(TypedDict):
    note_id: int
    task_result: str | list[Annotation]
    task_executor: str
    note_text: str


class CleanedDataEntry(TypedDict):
    note_id: int
    task_result: list[Annotation]
    task_executor: str
    note_text: str


class Token(TypedDict):
    token: str
    start_offset: int
    end_offset: int


# ------------------------------------------------------------------------------------------------------------------

def load_csv(csv_path: str) -> list[CleanedDataEntry]:
    data: list[CsvDataEntry] = pd.read_csv(csv_path).to_dict("records")
    with open("./data/label_mapping.json", encoding='utf8') as label_mapping_file:
        label_mapping: dict[str, str] = json.load(label_mapping_file)

    for data_entry in data:

        data_entry['task_result'] = ast.literal_eval(data_entry['task_result'])['entities']
        data_entry['task_result'].sort(key=lambda e: e['start_position'])

        for entity_annotation in data_entry['task_result']:
            entity_annotation['concept_category'] = label_mapping[str(entity_annotation['concept_category'])]

    return cast(list[CleanedDataEntry], data)


def generate_token_list(text: str) -> list[Token]:
    str_tokens = text.split()
    tokens = list()

    current_offset = 0

    for str_token in str_tokens:
        end_offset = current_offset + len(str_token)
        tokens.append(
            Token(
                token=str_token,
                start_offset=current_offset,
                end_offset=end_offset
            )
        )
        current_offset = end_offset + 1

    return tokens


def get_annotation_tokens(annotation: Annotation, tokens: list[Token], search_start_token: int = 0) -> tuple[int, list[Token]]:
    token_index = search_start_token
    annotated_tokens: list[Token] = list()

    for token_index, token_annotation in enumerate(tokens[search_start_token:], start=search_start_token):
        if annotation['end_position'] <= token_annotation['start_offset']:
            break
        elif (token_annotation['start_offset'] <= annotation['start_position'] < token_annotation['end_offset']) or (annotation['start_position'] <= token_annotation['start_offset'] < annotation['end_position']):
            annotated_tokens.append(token_annotation)

    return token_index, annotated_tokens


def convert_to_conll(entry: CleanedDataEntry) -> list[str]:
    lines = [f"{entry['note_id']} {entry['task_executor']} O\n"]

    tokens = generate_token_list(entry['note_text'])
    current_token_index = 0

    for annotation in entry['task_result']:
        current_token_index, annotated_tokens = get_annotation_tokens(annotation, tokens, current_token_index)

        if not annotated_tokens:
            raise AssertionError()

        for index, token in enumerate(annotated_tokens):
            lines.append(f"{token['token']} {token['start_offset']}-{token['end_offset']} {'B' if index == 0 else 'I'}-{annotation['concept_category']}")

    return lines


def main(csv_path: str):
    data = load_csv(csv_path)

    with open('./data/annotations_codiesp.json', 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open("./data/annotations_codiesp.conll", "w", encoding="utf8") as conll_file:
        conll_file.write("\n\n\n".join(["\n".join(convert_to_conll(entry)) for entry in data]))


if __name__ == '__main__':
    main('./data/annotations_codiesp.csv')
