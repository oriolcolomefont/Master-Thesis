import os
import msaf
import seaborn as sns
import numpy as np
from tqdm import tqdm
import mir_eval
import pandas as pd
from features import Embeddiogram

sns.set(style="dark")

AUDIO_DIR = "/home/jupyter/Master-Thesis/datasets/SALAMI/audio"
ANNOTATIONS_DIR = "/home/jupyter/Master-Thesis/datasets/SALAMI/references"
FEATURE = "embeddiogram"
BOUNDARIES_ID = msaf.config.default_bound_id
LABELS_ID = msaf.config.default_label_id
EVAL_WINDOW = 0.5  # The maximum allowed deviation for a correct boundary (in seconds)


def get_audio_and_annot_files(audio_dir, annotations_dir):
    audio_files_dict = {
        os.path.splitext(f)[0]: os.path.join(audio_dir, f)
        for f in os.listdir(audio_dir)
        if f.endswith(".mp3")
    }
    annot_files_dict = {
        os.path.splitext(os.path.basename(f))[0]: os.path.join(annotations_dir, f)
        for f in os.listdir(annotations_dir)
        if f.endswith(".jams")
    }

    valid_pairs = [
        (audio_files_dict[key], annot_files_dict[key])
        for key in audio_files_dict
        if key in annot_files_dict
    ]
    return zip(*valid_pairs)


def get_config(feature, boundaries_id, labels_id):
    return msaf.io.get_configuration(
        feature,
        annot_beats=False,
        framesync=False,
        boundaries_id=boundaries_id,
        labels_id=labels_id,
    )


def process_audio_file(audio_file_path, annot_file_path, feature_name):
    annot_file_path = annot_file_path[:-1]

    try:
        annot_intervals, annot_labels = msaf.io.read_references(annot_file_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    boundaries, labels = msaf.process(
        in_path=audio_file_path,
        feature=feature_name,
        plot=False,
    )

    return boundaries, labels, annot_intervals, annot_labels


def evaluate_segmentation(
    boundaries, labels, annot_intervals, annot_labels, eval_window
):
    # Evaluate boundary detection
    p, r, f = mir_eval.segment.detection(
        reference_intervals=annot_intervals,
        estimated_intervals=boundaries,
        window=eval_window,
    )
    ref_est, est_ref = mir_eval.segment.deviation(
        reference_intervals=annot_intervals, estimated_intervals=boundaries
    )

    # Evaluate segmentation
    scores = mir_eval.segment.evaluate(
        ref_intervals=annot_intervals,
        ref_labels=annot_labels,
        est_intervals=boundaries,
        est_labels=labels,
    )

    dict_1 = {
        "boundary_precision": p,
        "boundary_recall": r,
        "boundary_f_score": f,
        "reference_to_estimated": ref_est,
        "estimated_to_reference": est_ref,
    }

    return {**dict_1, **scores}


def boundaries_to_intervals(boundaries):
    intervals = np.zeros((len(boundaries) - 1, 2))
    intervals[:, 0] = boundaries[:-1]
    intervals[:, 1] = boundaries[1:]
    return intervals


def main():
    audio_files, annot_files = get_audio_and_annot_files(AUDIO_DIR, ANNOTATIONS_DIR)

    evaluations = []
    total_files = len(audio_files)
    save_interval = (
        total_files // 10
    )  # Save the CSV file after every 10% of the total iterations

    for i, (audio_file, annot_file) in enumerate(
        tqdm(zip(audio_files, annot_files), total=total_files), start=1
    ):
        print(f"Pair {i}:")
        print(f"Audio file: {audio_file}")
        print(f"Annotation file: {annot_file}")

        try:
            results = process_audio_file(
                audio_file_path=audio_file,
                annot_file_path=annot_file,
                feature_name=FEATURE,
            )

            # Convert boundaries to intervals
            estimated_intervals = boundaries_to_intervals(results[0])
            reference_intervals = boundaries_to_intervals(results[2])

            # Evaluate segmentation
            evaluation_results = evaluate_segmentation(
                estimated_intervals,
                results[1],
                reference_intervals,
                results[3],
                EVAL_WINDOW,
            )
            print(evaluation_results)

            evaluations.append(evaluation_results)
        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")
            continue

        if i % save_interval == 0:
            df = pd.DataFrame(evaluations)
            print(df.mean())
            df.to_csv(
                f"evaluation_results_{FEATURE}_{BOUNDARIES_ID}_{LABELS_ID}.csv",
                index=False,
                mode="w",
            )  # Overwrite the CSV file with each update


if __name__ == "__main__":
    main()
