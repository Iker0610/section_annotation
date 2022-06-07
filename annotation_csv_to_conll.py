import ast
import json
from typing import TypedDict, cast

import pandas as pd


class EntryAnnotation(TypedDict):
    id: str
    entity: str
    start_position: int
    end_position: int
    concept_category: str


class CsvDataEntry(TypedDict):
    note_id: int
    task_result: str | list[EntryAnnotation]
    task_executor: str
    note_text: str


class CleanedDataEntry(TypedDict):
    note_id: int
    task_result: list[EntryAnnotation]
    task_executor: str
    note_text: str


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


def get_annotation_tokens(annotation, tokens) -> list[tuple[str, int, int]]:
    # TODO
    pass


def convert_to_conll(entry: CleanedDataEntry):
    lines = [f"{entry['note_id']} {entry['task_executor']} 0"]

    tokens = entry['note_text'].split()
    for annotation in entry['task_result']:
        annotated_tokens = get_annotation_tokens(annotation, tokens)
        for token, start_offset, end_offset in annotated_tokens:
            lines.append(f"{token} {start_offset}-{end_offset} {annotation['concept_category']}")


def main(csv_path: str):
    data = load_csv(csv_path)

    with open('./data/annotations_codiesp.json', 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    files_conll = [convert_to_conll(entry) for entry in data]


if __name__ == '__main__':
    main('./data/annotations_codiesp.csv')
