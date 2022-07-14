import json
import re
from collections import deque
from pathlib import Path
from typing import TypedDict, cast, Callable, TextIO, Literal

# ------------------------------------------------------------------------------------------------------------------

token_in_parenthesis_matcher = re.compile(r"^[(\[{].*[)\]}][.,:;]?$")

start_delimiter_matcher = re.compile(r"^[(\[{¿¡]$")
start_delimiter_splitter = re.compile(r"(?<=.)([(\[{¡¿])")

end_delimiter_matcher = re.compile(r"^[)\]}?!][.,:;]?$")
end_delimiter_splitter = re.compile(r"([)\]}?!][.,:;]?)(?=[^.,:;])")

dot_splitter = re.compile(r"(?<!\.)(\.)(?=[A-ZÀ-ÖØ-ß])(?![A-ZÀ-ÖØ-ß]\.)|(?<=[a-zß-öø-ÿ])(\.)(?=[A-ZÀ-ÖØ-ß])")
punctuation_splitter = re.compile(r"([,:;]).")

# ------------------------------------------------------------------------------------------------------------------

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
    best_annotator: str
    split: Split
    annotator_annotations: dict[str, NoteAnnotations]


class Token(TypedDict):
    token: str
    start_offset: int
    end_offset: int


class AnnotationBoundary(TypedDict):
    label: str
    start_token_index: int
    end_token_index: int


GroupedAnnotatedDataset = dict[str, NoteWithAnnotations]
Boundaries = list[list[str]]


class BoundaryDataset(TypedDict):
    segmentation_type: str
    boundary_format: str
    boundary_types: list[str]
    items: dict[str, dict[str, Boundaries]]


# ------------------------------------------------------------------------------------------------------------------

label_list = [
    "RAZON_CONSULTA",
    "DERIVACION_DE/A",
    "ANTECEDENTES_FAMILIARES",
    "ANTECEDENTES_PERSONALES",
    "EXPLORACION",
    "TRATAMIENTO",
    "EVOLUCION",
    # "DIAGNOSTICO_DIFERENCIAL",
    # "DIAGNOSTICO_FINAL"
]


# ------------------------------------------------------------------------------------------------------------------

def save_popleft(queue: deque):
    try:
        return queue.popleft()
    except IndexError:
        return None


# ------------------------------------------------------------------------------------------------------------------

def fuse_subtokens(tokens: list[str], fuse_condition: Callable[[str, str], bool]) -> list[str]:
    tokens = tokens.copy()

    index = 0
    while index < len(tokens) - 1:
        previous_subtoken, subtoken = tokens[index], tokens[index + 1]

        if fuse_condition(previous_subtoken, subtoken):
            tokens[index:index + 2] = [previous_subtoken + subtoken]
        else:
            index += 1

    return tokens


def subtokenize(token: str, token_start_offset: int) -> list[Token]:
    if token_in_parenthesis_matcher.match(token):
        return [
            Token(
                token=token,
                start_offset=token_start_offset,
                end_offset=token_start_offset + len(token)
            )
        ]
    else:
        # Split token at end of parenthesis (keep punctuation marks stuck to parenthesis end)
        subtokens = end_delimiter_splitter.split(token)
        subtokens = fuse_subtokens(subtokens, lambda _, subtoken: bool(end_delimiter_matcher.match(subtoken)))

        # Split token at start of parenthesis
        subtokens = [split_subtoken for subtoken in subtokens for split_subtoken in start_delimiter_splitter.split(subtoken)]
        subtokens = fuse_subtokens(subtokens, lambda subtoken, _: bool(start_delimiter_matcher.match(subtoken)))

        # Split by punctuation mark but dot
        subtokens = [split_subtoken for subtoken in subtokens for split_subtoken in punctuation_splitter.split(subtoken)]
        subtokens = fuse_subtokens(subtokens, lambda _, subtoken: bool(subtoken in ',:;'))

        # Split by dot
        subtokens = [split_subtoken for subtoken in subtokens for split_subtoken in dot_splitter.split(subtoken) if split_subtoken]
        subtokens = fuse_subtokens(subtokens, lambda _, subtoken: bool(subtoken == '.'))

        # Convert strings to Token instances
        current_offset: int = token_start_offset
        for index, subtoken in enumerate(subtokens):
            end_offset = current_offset + len(subtoken)
            subtokens[index] = Token(
                token=subtoken,
                start_offset=current_offset,
                end_offset=end_offset
            )
            current_offset = end_offset

        return cast(list[Token], subtokens)


def generate_token_list(text: str) -> list[Token]:
    str_tokens = [token for token in re.split(r'(\r?\n| |\r\n?)', text) if token]
    tokens = list()

    current_offset = 0

    for str_token in str_tokens:
        end_offset = current_offset + len(str_token)

        if str_token and str_token not in ['\r\n', '\n', '\r', ' ']:
            tokens += subtokenize(str_token, current_offset)
        elif str_token in ['\r\n', '\n', '\r']:
            tokens.append(Token(
                token='<#EOL#>',
                start_offset=current_offset,
                end_offset=end_offset
            ))

        current_offset = end_offset

    return tokens


# ------------------------------------------------------------------------------------------------------------------

def get_annotation_boundaries(tokenized_text: list[Token], annotations: list[Annotation]) -> Boundaries:
    annotations = deque(annotations)
    annotation_boundaries: list[list[str]] = list()
    next_annotation = save_popleft(annotations)

    for token_index, token in enumerate(tokenized_text):
        if next_annotation is not None and next_annotation['start_position'] <= token['start_offset']:
            annotation_boundaries.append([next_annotation['label']])
            next_annotation = save_popleft(annotations)
        else:
            annotation_boundaries.append([])

    return annotation_boundaries


def generate_conll(filename: str, annotator: str, tokenized_text, boundaries: Boundaries, conll_file: TextIO):
    assert len(tokenized_text) == len(boundaries), "Tokenized text size and it's boundaries do not match"
    conll_file.write(f'{filename} {annotator} -\n\n')

    current_label = 'O'
    for token, label in zip(tokenized_text, boundaries):
        if label:
            current_label = label[0]

        conll_file.write(f"{token['token']} {token['start_offset']}-{token['end_offset']} {'B' if label else 'I'}-{current_label}\n")

    conll_file.write('\n\n')


# ------------------------------------------------------------------------------------------------------------------

def main(json_path: str, conll_output_path: str, boundary_dataset_path: str):
    with open(json_path, encoding='utf8') as json_file:
        dataset: GroupedAnnotatedDataset = json.load(json_file)['annotated_dataset']

    boundary_dataset = BoundaryDataset(
        segmentation_type='linear',
        boundary_format='sets',
        boundary_types=label_list,
        items=dict()
    )
    conll_output_path = Path(conll_output_path)

    with \
            open(conll_output_path.with_suffix('.train.conll'), "w", encoding="utf8") as train_conll_file, \
            open(conll_output_path.with_suffix('.dev.conll'), "w", encoding="utf8") as dev_conll_file, \
            open(conll_output_path.with_suffix('.test.conll'), "w", encoding="utf8") as test_conll_file:

        file_handler_mapper = {
            "train": train_conll_file,
            "dev": dev_conll_file,
            "test": test_conll_file,
        }

        for filename, annotated_file_data in dataset.items():
            note_text: str = annotated_file_data['note_text']
            tokenized_note_text: list[Token] = generate_token_list(note_text)

            boundaries_per_annotator: dict[str: Boundaries] = {
                annotator: get_annotation_boundaries(tokenized_note_text, annotations['annotations'])
                for annotator, annotations in annotated_file_data['annotator_annotations'].items()
            }

            if 1 < len(boundaries_per_annotator):
                boundary_dataset['items'][filename] = boundaries_per_annotator

            generate_conll(
                filename,
                annotated_file_data['best_annotator'],
                tokenized_note_text,
                boundaries_per_annotator[annotated_file_data['best_annotator']],
                file_handler_mapper[annotated_file_data['split']]
            )

    with open(boundary_dataset_path, "w", encoding="utf8") as boundary_dataset_file:
        json.dump(boundary_dataset, boundary_dataset_file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main(
        json_path='./data/annotations_codiesp_splitted.json',
        conll_output_path='./data/conll/annotations_codiesp.conll',
        boundary_dataset_path='./data/annotations_codiesp_boundaries_dataset.json'
    )
