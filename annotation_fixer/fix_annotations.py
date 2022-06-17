import json
import logging
from collections import deque
from typing import TypedDict

import pandas as pd

logger = logging.getLogger()


def save_popleft(queue: deque):
    try:
        return queue.popleft()
    except IndexError:
        return None


class Annotation(TypedDict):
    id: str
    entity: str
    start_position: int
    end_position: int
    concept_category: str


class DataEntry(TypedDict):
    note_id: int
    task_result: list[Annotation]
    task_executor: str
    note_text_corrupted: str
    note_text_golden: str


def fix_annotation(annotated_text: DataEntry):
    annotations = deque(annotated_text['task_result'])

    current_annotation: Annotation | None = None
    next_annotation: Annotation | None = save_popleft(annotations)

    if next_annotation is None:
        logger.error(f"File {annotated_text['note_id']} has no annotations")
        return

    golden_text = deque(annotated_text['note_text_golden'])

    current_corrupted_offset = 0
    current_golden_offset = 0
    previous_offset_adjustment = 0
    offset_adjustment = 0

    for char in annotated_text['note_text_corrupted']:
        # Advance in golden text while character do not match
        while char != golden_text.popleft():
            current_golden_offset += 1
            offset_adjustment += 1

        # Comprobamos si hemos cambiado de anotación
        if next_annotation['start_position'] <= current_corrupted_offset:
            # Si había una previa corregimos su offset de fin
            if current_annotation is not None:
                current_annotation['end_position'] = current_golden_offset
                current_annotation['start_position'] = current_annotation['start_position'] + previous_offset_adjustment
                current_annotation['entity'] = annotated_text['note_text_golden'][current_annotation['start_position']:current_annotation['end_position']]

            # Actualizamos la anotación actual
            current_annotation = next_annotation
            next_annotation = save_popleft(annotations)
            previous_offset_adjustment = offset_adjustment

        # Si no quedan más anotaciones extendemos la última hasta el final
        if next_annotation is None:
            current_annotation['end_position'] = len(annotated_text['note_text_golden'])
            current_annotation['start_position'] = current_annotation['start_position'] + previous_offset_adjustment
            current_annotation['entity'] = annotated_text['note_text_golden'][current_annotation['start_position']:current_annotation['end_position']]
            break

        # Update offsets
        current_corrupted_offset += 1
        current_golden_offset += 1


def main2():
    data: list[DataEntry] = [
        {
            "note_id": 9223233032138632789,
            "task_result": [
                {
                    "id": "new-lo9ca8vsf",
                    "entity": "Niño de 3 meses con lesiones eritematosas redondeadas de bordes definidos con centro más pálido en el rostro los miembros superiores de aspecto reticulado en miembros inferiores y dorso de 1 mes de evolución. Hepatoesplenomegalia",
                    "end_position": 229,
                    "start_position": 0,
                    "concept_category": "RAZON_CONSULTA"
                },
                {
                    "id": "new-m8igjh1hb",
                    "entity": "Madre y abuela materna con diagnóstico de Lupus Eritematoso sistémico (LES)",
                    "end_position": 305,
                    "start_position": 230,
                    "concept_category": "ANTECEDENTES_FAMILIARES"
                },
                {
                    "id": "new-zkoy6gx6y",
                    "entity": "Exámenes complementarios: Hb 10g/dl Hto 30% GPT 78 UI/L C3 53 mg/dl y C4 3mg/dl FAN+ 1/100 acs. anti Ro anti La anti Sm y anti U1RNP negativos.Estudio histopatológico: hiperqueratosis atrofia epidérmica vacuolización de queratinocitos basales e infiltrado difuso mononuclear en dermis superficial. Inmunofluorescencia directa (IFD): banda segmentaria dermoepidérmica de disposición granular fina IgG+ y C3+.Valoración cardiológica normal",
                    "end_position": 742,
                    "start_position": 305,
                    "concept_category": "EXPLORACION"
                },
                {
                    "id": "new-sgxkaylis",
                    "entity": "Tratamiento: hidrocortisona 1% crema.",
                    "end_position": 780,
                    "start_position": 743,
                    "concept_category": "TRATAMIENTO"
                },
                {
                    "id": "new-lkigdyw2t",
                    "entity": "Resolución de las lesiones cutáneas negativización del FAN y normalización del hepatograma a los 6 meses de vida",
                    "end_position": 892,
                    "start_position": 780,
                    "concept_category": "EVOLUCION"
                }
            ],
            "task_executor": "u1755",
            "note_text_corrupted": "Niño de 3 meses con lesiones eritematosas redondeadas de bordes definidos con centro más pálido en el rostro los miembros superiores de aspecto reticulado en miembros inferiores y dorso de 1 mes de evolución. Hepatoesplenomegalia.Madre y abuela materna con diagnóstico de Lupus Eritematoso sistémico (LES)Exámenes complementarios: Hb 10g/dl Hto 30% GPT 78 UI/L C3 53 mg/dl y C4 3mg/dl FAN+ 1/100 acs. anti Ro anti La anti Sm y anti U1RNP negativos.Estudio histopatológico: hiperqueratosis atrofia epidérmica vacuolización de queratinocitos basales e infiltrado difuso mononuclear en dermis superficial. Inmunofluorescencia directa (IFD): banda segmentaria dermoepidérmica de disposición granular fina IgG+ y C3+.Valoración cardiológica normal.Tratamiento: hidrocortisona 1% crema.Resolución de las lesiones cutáneas negativización del FAN y normalización del hepatograma a los 6 meses de vida.",
            "note_text_golden": 'Niño de 3 meses con lesiones eritematosas, redondeadas, de bordes definidos, con centro más pálido en el rostro, los miembros superiores, de aspecto reticulado en miembros inferiores y dorso de 1 mes de evolución. Hepatoesplenomegalia.\n \n Madre y abuela materna con diagnóstico de Lupus Eritematoso sistémico (LES)\n Exámenes complementarios: Hb 10g/dl, Hto 30%, GPT 78 UI/L, C3 53 mg/dl y C4 3mg/dl, FAN+ 1/100, acs. anti Ro, anti La, anti Sm y anti U1RNP negativos.\n Estudio histopatológico: hiperqueratosis, atrofia epidérmica, vacuolización de queratinocitos basales e infiltrado difuso mononuclear en dermis superficial. Inmunofluorescencia directa (IFD): banda segmentaria dermoepidérmica de disposición granular fina IgG+ y C3+.\n Valoración cardiológica normal.\n Tratamiento: hidrocortisona 1% crema.\n Resolución de las lesiones cutáneas, negativización del FAN y normalización del hepatograma a los 6 meses de vida.\n'
        },
    ]

    for annotated_text in data:
        fix_annotation(annotated_text)

    with open('./fixed_annotation_codiesp.json', 'w', encoding='utf8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=2)


def main(csv_path: str):
    data = pd.read_csv(csv_path)
    # TODO Esperar a tener una forma de juntar los 2 CSV


if __name__ == '__main__':
    # path_to_file = "./notes_mapped.csv"
    # main(path_to_file)
    main2()
