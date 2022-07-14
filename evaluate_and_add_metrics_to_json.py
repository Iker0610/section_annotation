import json
import math
from collections import defaultdict
from typing import TypedDict

from segeval.agreement import actual_agreement_linear
from segeval.agreement.bias import artstein_poesio_bias_linear
from segeval.agreement.kappa import fleiss_kappa_linear
from segeval.agreement.pi import fleiss_pi_linear
from segeval.data import Dataset
from segeval.data.jsonutils import input_linear_boundaries_json
from segeval.similarity import weight_a, boundary_statistics
from segeval.similarity.boundary import boundary_similarity
from segeval.similarity.weight import weight_s, weight_t


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


# ------------------------------------------------------------------------------------------------------------------

def calculate_dataset_intertagger_metrics(metric_function, **kwargs):
    return DatasetMetrics(
        KAPPA=float(fleiss_kappa_linear(fnc_compare=metric_function, **kwargs)),
        PI=float(fleiss_pi_linear(fnc_compare=metric_function, **kwargs)),
        Agreement=float(actual_agreement_linear(fnc_compare=metric_function, **kwargs)),
        BIAS=float(artstein_poesio_bias_linear(fnc_compare=metric_function, **kwargs)),
    )


def weighted_A(additions, *args, **kwargs):
    additions = len(additions)

    if additions < 3:
        return 0.55 * additions
    else:
        return additions


def weighted_A2(additions, *args, **kwargs):
    additions = len(additions)

    if additions:
        # return 1 + ((math.log2((additions / 3) + 0.125)) / 1.85 * additions)
        # return additions * (math.log10(additions) / 1.25 + 0.25)
        return additions * ((math.log(additions - 0.9, 50) / 2) + 0.8)
    else:
        return 0


def weighted_A3(additions, *args, **kwargs):
    additions = len(additions)

    def weight(x):
        return 0.75 + (math.tanh((x - 1.5) - 2) / 4)

    return additions * weight(additions) if additions else 0


def weight_t_scale2(transpositions, max_n):
    """
    Default weighting function for transposition edit operations by the distance that transpositions span.
    """

    def weight(x):
        return 0.35 + (math.tanh(x / 10) / 3)

    numerator = 0
    for transposition in transpositions:
        num_tokens_moved = abs(transposition[0] - transposition[1])
        numerator += 0 if num_tokens_moved <= 2 else weight(num_tokens_moved - 15)
    return numerator


def calculate_metrics(dataset: Dataset):
    weight_functions = (weight_a, weight_s, weight_t)
    weight_functions2 = (
        weighted_A3,
        lambda *args, **kwargs: 1.3 * weight_s(*args, **kwargs),
        # lambda *args, **kwargs: float(weight_t_scale(*args, **kwargs))
        weight_t_scale2
    )
    parameters_B = {
        'dataset': dataset,
        'weight': weight_functions,
    }

    parameters_B2 = {
        'dataset': dataset,
        'weight': weight_functions2,
        'n_t': 40
    }

    metrics: dict[str, dict] = {
        'intertagger_metrics': {
            'B': calculate_dataset_intertagger_metrics(metric_function=boundary_similarity, **parameters_B),
            'S': calculate_dataset_intertagger_metrics(metric_function=boundary_similarity, **parameters_B2),
        },
        'file_stats': defaultdict(list),
    }

    stats_per_file: dict = boundary_statistics(**parameters_B2)

    b = boundary_similarity(**parameters_B)
    b2 = boundary_similarity(**parameters_B2)

    for file_code, file_stats in stats_per_file.items():
        filename, *annotators = file_code.split(',')

        del file_stats['boundaries_all']
        del file_stats['boundary_types']
        del file_stats['full_misses']
        del file_stats['matches']

        metrics['file_stats'][filename].append({
            'annotators': annotators,
            'stats': file_stats,
            'metrics': {
                'S': b2[file_code],
                'B': b[file_code]
            }
        })

    return metrics


def evaluate_dataset(dataset: DatasetWithMetrics, boundary_dataset: Dataset):
    metrics = calculate_metrics(boundary_dataset)
    dataset['dataset_metrics'] = metrics['intertagger_metrics']
    for filename, file_data in dataset['annotated_dataset'].items():
        file_data['metrics'] = metrics['file_stats'].setdefault(filename, list())

    return dataset


# ------------------------------------------------------------------------------------------------------------------

def main(dataset_json_path: str, boundary_dataset_path: str, output_json_path: str):
    boundary_dataset = input_linear_boundaries_json(boundary_dataset_path)

    with open(dataset_json_path, encoding='utf-8') as json_file:
        dataset = json.load(json_file)

    evaluated_dataset = evaluate_dataset(dataset, boundary_dataset)

    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(evaluated_dataset, json_file, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main(
        './data/annotations_codiesp.json',
        './data/annotations_codiesp_boundaries_dataset.json',
        './data/annotations_codiesp_evaluated.json',
    )
