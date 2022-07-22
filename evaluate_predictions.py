import argparse
import json
import statistics
from typing import TypedDict, Optional

from segeval.format import BoundaryFormat
from segeval.similarity import boundary_statistics, B2_parameters as b2_default_parameters
from segeval.similarity.boundary import boundary_similarity
from segeval.similarity.weight import weight_s, weight_t, weight_a


class Annotation(TypedDict):
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
    annotator: str
    annotator_annotations: dict[str, NoteAnnotations]


GroupedAnnotatedDataset = dict[str, NoteWithAnnotations]


class DatasetMetrics(TypedDict):
    arithmetic_mean: float
    standard_deviation: float


class EvaluatedDataset(TypedDict):
    dataset_metrics: DatasetMetrics
    annotated_dataset: GroupedAnnotatedDataset


# --------------------------------------------------------------------
label_set = {
    "RAZON_CONSULTA",
    "DERIVACION_DE/A",
    "ANTECEDENTES_FAMILIARES",
    "ANTECEDENTES_PERSONALES",
    "EXPLORACION",
    "TRATAMIENTO",
    "EVOLUCION",
    # "DIAGNOSTICO_DIFERENCIAL",
    # "DIAGNOSTICO_FINAL"
}


# --------------------------------------------------------------------
def evaluate_file(label_boundaries: list[frozenset], prediction_boundaries: list[frozenset], annotator: str) -> tuple[dict, float]:
    parameters_B = {'boundary_types': label_set, 'boundary_format': BoundaryFormat.sets, 'weight': (weight_a, weight_s, weight_t)}
    parameters_B2 = parameters_B | b2_default_parameters

    file_stats: dict = boundary_statistics(label_boundaries, prediction_boundaries, **parameters_B2)
    del file_stats['boundaries_all']
    del file_stats['boundary_types']
    del file_stats['full_misses']
    del file_stats['matches']

    b = boundary_similarity(reference=label_boundaries, hypothesis=prediction_boundaries, **parameters_B)
    b2 = boundary_similarity(reference=label_boundaries, hypothesis=prediction_boundaries, **parameters_B2)

    return {'annotators': [annotator, 'model'],
            'stats': file_stats,
            'metrics': {'B2': b2, 'B': b}
            }, b2


def convert_and_evaluate_file(note_id: int, file_name: str, annotator: str, file_lines: list[list[str, str, str, str]], note_text: str) -> tuple[NoteWithAnnotations, float]:
    note_with_annotations: NoteWithAnnotations = {
        'note_id': note_id,
        'filename': file_name,
        'note_text': note_text,
        'metrics': list(),
        'annotator': annotator,
        'annotator_annotations': {
            'gold': {
                'annotator': annotator,
                'annotations': list()
            },
            'prediction': {
                'annotator': 'model',
                'annotations': list()
            }
        }
    }

    # -------------------------------------------------------------------
    # Generate JSON and Boundaries

    labels_boundaries: list[frozenset] = []
    predictions_boundaries: list[frozenset] = []

    current_label: Optional[str] = None
    current_prediction: Optional[str] = None

    current_label_start_offset = 0
    current_prediction_start_offset = 0

    for line in file_lines:
        _, offset, label, prediction = line
        label: str = label.split('-')[-1]
        prediction: str = prediction.split('-')[-1]

        start_offset: int = int(offset.split('-')[0])

        if label != current_label:
            if current_label:
                note_with_annotations['annotator_annotations']['gold']['annotations'].append(
                    {
                        'entity': note_text[current_label_start_offset:start_offset],
                        'start_position': current_label_start_offset,
                        'end_position': start_offset,
                        'label': current_label,
                    }
                )
            current_label = label
            current_label_start_offset = start_offset
            labels_boundaries.append(frozenset([label]))
        else:
            labels_boundaries.append(frozenset())

        if prediction != current_prediction:
            if current_prediction:
                note_with_annotations['annotator_annotations']['prediction']['annotations'].append(
                    {
                        'entity': note_text[current_prediction_start_offset:start_offset],
                        'start_position': current_prediction_start_offset,
                        'end_position': start_offset,
                        'label': current_prediction,
                    }
                )
            current_prediction = prediction
            current_prediction_start_offset = start_offset
            predictions_boundaries.append(frozenset([prediction]))
        else:
            predictions_boundaries.append(frozenset())
    # Add last annotation:
    if current_label:
        note_with_annotations['annotator_annotations']['gold']['annotations'].append(
            {
                'entity': note_text[current_label_start_offset:],
                'start_position': current_label_start_offset,
                'end_position': len(note_text),
                'label': current_label,
            }
        )

    if current_prediction:
        note_with_annotations['annotator_annotations']['prediction']['annotations'].append(
            {
                'entity': note_text[current_prediction_start_offset:],
                'start_position': current_prediction_start_offset,
                'end_position': len(note_text),
                'label': current_prediction,
            }
        )

    # -------------------------------------------------------------------

    # Evaluate
    metric_dict, metric = evaluate_file(labels_boundaries, predictions_boundaries, annotator)
    note_with_annotations['metrics'].append(metric_dict)

    # -------------------------------------------------------------------

    return note_with_annotations, metric


def convert_to_json_and_evaluate(input_file: list[str], text_mapper: dict[str, str]):
    json_dataset: EvaluatedDataset = {
        'dataset_metrics': dict(),
        'annotated_dataset': dict()
    }

    current_file_name = None
    current_file_annotator = None
    current_note_id = -1
    current_file_lines: list[list[str, str, str, str]] = []
    metric_list = []

    for line in input_file:
        if not line:
            continue

        line_columns = line.split(' ')

        # Si es un inicio de fichero
        if line_columns[-1] == '-':
            current_note_id += 1

            # Si había un fichero previo lo procesamos
            if current_file_name and current_file_annotator and current_file_lines:
                converted_file_data, metric = convert_and_evaluate_file(current_note_id, current_file_name, current_file_annotator, current_file_lines, text_mapper[current_file_name])
                json_dataset['annotated_dataset'][current_file_name] = converted_file_data
                metric_list.append(metric)

            # Reseteamos los valores
            if len(file_name_annotator := line_columns[1].split('#')) < 2:
                file_name_annotator.append('gold')

            current_file_name, current_file_annotator = file_name_annotator
            current_file_lines = []

        # Si no lo es añadimos los datos a la lista
        else:
            current_file_lines.append(line_columns)

    # Procesamos el último fichero
    if current_file_name and current_file_annotator and current_file_lines:
        converted_file_data, metric = convert_and_evaluate_file(current_note_id, current_file_name, current_file_annotator, current_file_lines, text_mapper[current_file_name])
        json_dataset['annotated_dataset'][current_file_name] = converted_file_data
        metric_list.append(metric)

    json_dataset['dataset_metrics']['arithmetic_mean'] = statistics.mean(metric_list)
    json_dataset['dataset_metrics']['standard_deviation'] = statistics.stdev(metric_list)

    return json_dataset


def convert_to_json_and_evaluate_from_file(input_file: str, text_mapper: str, output_file: str):
    with open(input_file, encoding='utf-8') as f:
        input_file = f.read().splitlines()

    with open(text_mapper, encoding='utf-8') as f:
        text_mapper = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_to_json_and_evaluate(input_file, text_mapper), f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', type=str, default='./predictions/eval_predictions.conll', help='CONLL file with tokens, labels and predictions.')
    parser.add_argument('-t', '--text_mapper', type=str, default='./mappers/filename_to_note_text.json', help='CONLL file with tokens, labels and predictions.')
    parser.add_argument('-o', '--output_file', type=str, default='./predictions/eval_predictions.json', help='Output JSON file with best annotator selected and split assigned.')

    args = parser.parse_args()

    convert_to_json_and_evaluate_from_file(**vars(args))
