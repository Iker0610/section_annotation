import ast
import json
import logging
from collections import deque
from typing import TypedDict, cast

import pandas as pd

# ------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger()


def save_popleft(queue: deque):
    try:
        return queue.popleft()
    except IndexError:
        return None


# ------------------------------------------------------------------------------------------------------------------

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


class CorruptedDataEntry(TypedDict):
    note_id: int
    task_result: list[Annotation]
    task_executor: str
    note_text: str


class FixedDataEntry(TypedDict):
    note_id: int
    filename: str
    annotations: list[Annotation]
    annotator: str
    note_text: str


class CorruptedNoteData(TypedDict):
    filename: str
    original_note: str


GroupedAnnotatedDataset = dict[str, dict[str, FixedDataEntry]]


# ------------------------------------------------------------------------------------------------------------------

def load_data_csv(csv_path: str) -> list[CorruptedDataEntry]:
    data: list[CsvDataEntry] = pd.read_csv(csv_path).to_dict("records")
    with open("./mappers/label_mapping.json", encoding='utf8') as label_mapping_file:
        label_mapping: dict[str, str] = json.load(label_mapping_file)

    for data_entry in data:

        data_entry['task_result'] = ast.literal_eval(data_entry['task_result'])['entities']
        data_entry['task_result'].sort(key=lambda e: e['start_position'])

        for entity_annotation in data_entry['task_result']:
            entity_annotation['concept_category'] = label_mapping[str(entity_annotation['concept_category'])]

    return cast(list[CorruptedDataEntry], data)


# ------------------------------------------------------------------------------------------------------------------


def fix_corrupted_annotation(corrupted_data_entry: CorruptedDataEntry, corrupted_corpus_mapper: dict) -> FixedDataEntry:
    annotated_note_data: CorruptedNoteData = corrupted_corpus_mapper[corrupted_data_entry['note_text']]
    original_note, filename = annotated_note_data['original_note'], annotated_note_data['filename']

    annotations = deque(corrupted_data_entry['task_result'])

    current_annotation: Annotation | None = None
    next_annotation: Annotation | None = save_popleft(annotations)

    if next_annotation is None:
        logger.error(f"File {corrupted_data_entry['note_id']} has no annotations")
        return FixedDataEntry(
            note_id=corrupted_data_entry['note_id'],
            filename=filename,
            annotations=corrupted_data_entry['task_result'],
            annotator=corrupted_data_entry['task_executor'],
            note_text=original_note,
        )

    original_note_chars = deque(original_note)

    current_corrupted_offset = 0
    current_golden_offset = 0
    previous_offset_adjustment = 0
    offset_adjustment = 0

    for char in corrupted_data_entry['note_text']:
        # Advance in golden text while character do not match
        while char != original_note_chars.popleft():
            current_golden_offset += 1
            offset_adjustment += 1

        # Comprobamos si hemos cambiado de anotación
        if next_annotation['start_position'] <= current_corrupted_offset:
            # Si había una previa corregimos su offset de fin
            if current_annotation is not None:
                current_annotation['end_position'] = current_golden_offset
                current_annotation['start_position'] = current_annotation['start_position'] + previous_offset_adjustment
                current_annotation['entity'] = original_note[current_annotation['start_position']:current_annotation['end_position']]

            # Actualizamos la anotación actual
            current_annotation = next_annotation
            next_annotation = save_popleft(annotations)
            previous_offset_adjustment = offset_adjustment

        # Si no quedan más anotaciones extendemos la última hasta el final
        if next_annotation is None:
            current_annotation['end_position'] = len(original_note)
            current_annotation['start_position'] = current_annotation['start_position'] + previous_offset_adjustment
            current_annotation['entity'] = original_note[current_annotation['start_position']:current_annotation['end_position']]
            break

        # Update offsets
        current_corrupted_offset += 1
        current_golden_offset += 1

    return FixedDataEntry(
        note_id=corrupted_data_entry['note_id'],
        filename=filename,
        annotations=corrupted_data_entry['task_result'],
        annotator=corrupted_data_entry['task_executor'],
        note_text=original_note,
    )


def fix_corrupted_dataset(dataset: list[CorruptedDataEntry]) -> list[FixedDataEntry]:
    with open('./mappers/corrupted_notes_mapping.json', encoding='utf8') as mapper_file:
        corrupted_corpus_mapper: dict = json.load(mapper_file)

    return [fix_corrupted_annotation(corrupted_data_entry, corrupted_corpus_mapper) for corrupted_data_entry in dataset]


# ------------------------------------------------------------------------------------------------------------------

def group_dataset_annotations_by_file(dataset: list[FixedDataEntry]) -> GroupedAnnotatedDataset:
    grouped_dataset: GroupedAnnotatedDataset = dict()
    for annotated_text in dataset:
        grouped_dataset.setdefault(annotated_text['filename'], dict())[annotated_text['annotator']] = annotated_text
    return dict(sorted(grouped_dataset.items()))


# ------------------------------------------------------------------------------------------------------------------


def main(csv_path: str):
    raw_data = load_data_csv(csv_path)
    fixed_data = fix_corrupted_dataset(raw_data)
    grouped_data = group_dataset_annotations_by_file(fixed_data)

    with open('data/annotations_codiesp.json', 'w', encoding='utf8') as f:
        json.dump(grouped_data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main('./data/annotations_codiesp.csv')
