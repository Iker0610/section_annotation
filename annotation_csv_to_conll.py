import ast
import json
import re
from pprint import pprint
from typing import TypedDict, cast, Callable

import pandas as pd

token_in_parenthesis_matcher = re.compile(r"^[(\[{].*[)\]}][.,:;]?$")

start_delimiter_matcher = re.compile(r"^[(\[{¿¡]$")
start_delimiter_splitter = re.compile(r"(?<=.)([(\[{¡¿])")

end_delimiter_matcher = re.compile(r"^[)\]}?!][.,:;]?$")
end_delimiter_splitter = re.compile(r"([)\]}?!][.,:;]?)(?=[^.,:;])")

dot_splitter = re.compile(r"(?<!\.)(\.)(?=[A-ZÀ-ÖØ-ß])(?![A-ZÀ-ÖØ-ß]\.)|(?<=[a-zß-öø-ÿ])(\.)(?=[A-ZÀ-ÖØ-ß])")
punctuation_splitter = re.compile(r"([,:;]).")


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
    str_tokens = text.split()
    tokens = list()

    current_offset = 0

    for str_token in str_tokens:
        end_offset = current_offset + len(str_token)
        tokens += subtokenize(str_token, current_offset)
        current_offset = end_offset + 1

    return tokens


def convert_to_conll(entry: CleanedDataEntry) -> list[str]:
    lines = [f"{entry['note_id']} {entry['task_executor']} O\n"]

    tokens = generate_token_list(entry['note_text'])

    annotations: list[tuple[Annotation, Annotation | None]] = list(zip(entry['task_result'], entry['task_result'][1:] + [None]))

    current_annotation, next_annotation = annotations.pop(0)
    is_annotation_start = True

    for token in tokens:
        if next_annotation is not None and next_annotation['start_position'] <= token['start_offset']:
            current_annotation, next_annotation = annotations.pop(0)
            is_annotation_start = True

        lines.append(f"{token['token']} {token['start_offset']}-{token['end_offset']} {'B' if is_annotation_start else 'I'}-{current_annotation['concept_category']}")
        is_annotation_start = False
    return lines


def main(csv_path: str):
    data = load_csv(csv_path)

    with open('./data/annotations_codiesp.json', 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open("./data/annotations_codiesp.conll", "w", encoding="utf8") as conll_file:
        conll_file.write("\n\n\n".join(["\n".join(convert_to_conll(entry)) for entry in data]))


if __name__ == '__main__':
    main('./data/annotations_codiesp.csv')
