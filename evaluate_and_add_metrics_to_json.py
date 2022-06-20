import json
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
from segeval.similarity.segmentation import segmentation_similarity
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

def calculate_dataset_intertagger_metrics(dataset, metric_function, weight_functions):
    return DatasetMetrics(
        KAPPA=float(fleiss_kappa_linear(dataset=dataset, fnc_compare=metric_function, weight=weight_functions)),
        PI=float(fleiss_pi_linear(dataset=dataset, fnc_compare=metric_function, weight=weight_functions)),
        Agreement=float(actual_agreement_linear(dataset=dataset, fnc_compare=metric_function, weight=weight_functions)),
        BIAS=float(artstein_poesio_bias_linear(dataset=dataset, fnc_compare=metric_function, weight=weight_functions)),
    )


def calculate_metrics(dataset: Dataset):
    weight_functions = (weight_a, weight_s, weight_t)

    metrics: dict[str, dict] = {
        'intertagger_metrics': {
            'B': calculate_dataset_intertagger_metrics(dataset=dataset, metric_function=boundary_similarity, weight_functions=weight_functions),
            'S': calculate_dataset_intertagger_metrics(dataset=dataset, metric_function=segmentation_similarity, weight_functions=weight_functions),
        },
        'file_stats': defaultdict(list),
    }

    stats_per_file: dict = boundary_statistics(
        dataset=dataset,
        weight=weight_functions,
    )

    s = segmentation_similarity(
        dataset=dataset,
        weight=weight_functions,
    )

    b = boundary_similarity(
        dataset=dataset,
        weight=weight_functions,
    )

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
                'S': s[file_code],
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
