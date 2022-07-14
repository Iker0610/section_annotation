import argparse
import json
from typing import TypedDict, Literal
from sklearn.model_selection import train_test_split

Split = Literal['train', 'dev', 'test']

class Annotation(TypedDict):
    id: str
    entity: str
    start_position: int
    end_position: int
    label: str


class NoteAnnotations(TypedDict):
    annotator: str
    annotations: list[Annotation]


class NoteWithAnnotations(TypedDict):
    note_id: int
    filename: str
    note_text: str
    metrics: list
    best_annotator: str
    split: Split
    annotator_annotations: dict[str, NoteAnnotations]


GroupedAnnotatedDataset = dict[str, NoteWithAnnotations]


class DatasetMetrics(TypedDict):
    Agreement: float
    BIAS: float
    KAPPA: float
    PI: float


class DatasetWithMetrics(TypedDict):
    dataset_metrics: dict[str, DatasetMetrics]
    annotated_dataset: GroupedAnnotatedDataset


# -----------------------------------------------------------------------------------------------------------------------

annotator_priority = ['u1796', 'u1795', 'u1794', 'u942', 'u1720', 'u1755']
priority_annotators = {'u1796', 'u1795', 'u1794', 'u942'}


# -----------------------------------------------------------------------------------------------------------------------


def select_annotator(note: NoteWithAnnotations) -> str:
    for annotator in annotator_priority:
        annotation = note['annotator_annotations'].get(annotator)
        if annotation and annotation['annotations']:
            note['best_annotator'] = annotator
            return annotator

    raise ValueError()


def assign_annotator_and_split(input_file: str, output_file: str):
    # Load JSON
    with open(input_file, encoding='utf-8') as input_file_output:
        data: DatasetWithMetrics = json.load(input_file_output)

    files = []
    assigned_annotator_group: list[bool] = []

    for filename, file in data['annotated_dataset'].items():
        files.append(filename)
        assigned_annotator_group.append(select_annotator(file) in priority_annotators)

    # Split original dataframe into train and temp dataframes.
    train_files, temp_files, _, y_temp = train_test_split(files, assigned_annotator_group, stratify=assigned_annotator_group, train_size=0.75)

    # Split the temp dataframe into val and test dataframes.
    dev_files, test_files, _, _ = train_test_split(temp_files, y_temp, stratify=y_temp, test_size=0.6)

    train_files: set[str] = set(train_files)
    dev_files: set[str] = set(dev_files)
    test_files: set[str] = set(test_files)

    for filename, file in data['annotated_dataset'].items():
        if filename in test_files:
            file['split'] = 'test'
        elif filename in dev_files:
            file['split'] = 'dev'
        else:
            file['split']='train'
            if filename not in train_files:
                print(f"File {filename} was not innitialy assigned to any split so it's been assigned to train split.")

    with open(output_file, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, default='./data/annotations_codiesp.json', help='JSON file with annotations.')
    parser.add_argument('-o', '--output_file', type=str, default='./data/annotations_codiesp_splitted.json', help='Output JSON file with best annotator selected and split assigned.')

    args = parser.parse_args()

    assign_annotator_and_split(**vars(args))
