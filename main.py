import itertools
import os
import logging
import traceback
import train
import features
from msaf import config
import eval
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def create_settings():
    settings_dict = {
        "FILE_LIST_PATH": "./datasets/MSD/MSD_audio_limit=all.csv",
        "DATASET_NAME": "Million Song Dataset",
        "SAMPLE_RATE": config.sample_rate,
        "MAX_EPOCHS": 100,
        "EVAL_WINDOW": 0.5,
    }

    batch_sizes = sorted([8])
    window_sizes = sorted([i * settings_dict["SAMPLE_RATE"] for i in range(4, 8)])
    feature_list = sorted(["pcp", "mfcc", "embeddiogram"])
    clip_durations = sorted([3.0, 7.0, 15.0])

    settings = []

    for (
        batch_size,
        window_size,
        feature,
        clip_duration,
    ) in itertools.product(
        batch_sizes, window_sizes, feature_list, clip_durations
    ):
        settings_item = settings_dict.copy()
        settings_item.update(
            {
                "BATCH_SIZE": batch_size,
                "WINDOW_SIZE": window_size,
                "FEATURE": feature,
                "CLIP_DURATION": clip_duration,
            }
        )
        settings.append(settings_item)

    return settings


def main():
    settings = create_settings()

    for setting in tqdm(settings):
        logging.info(f"Processing settings: {setting}")
        try:

            """
            SETTING UP TRAINING PARAMETERS
            """

            train.FILE_LIST_PATH = setting["FILE_LIST_PATH"]
            train.DATASET_NAME = setting["DATASET_NAME"]
            train.BATCH_SIZE = setting["BATCH_SIZE"]
            train.CLIP_DURATION = setting["CLIP_DURATION"]
            train.SAMPLE_RATE = setting["SAMPLE_RATE"]
            train.MAX_EPOCHS = setting["MAX_EPOCHS"]

            """
            TRAINING THE MODEL
            """

            _, best_model_path = train.main()


            """
            SETTING UP TRAINING PARAMETERS
            """
            
            eval.FEATURE = setting["FEATURE"]
            eval.EVAL_WINDOW = setting["EVAL_WINDOW"]
            features.CKPT_PATH = best_model_path
            features.WINDOW_SIZE = setting["WINDOW_SIZE"]

            # Check if a file exists before attempting to delete it
            if os.path.exists("/home/jupyter/oriol/Master-Thesis/.features_msaf_tmp.json"):
                os.remove("/home/jupyter/oriol/Master-Thesis/.features_msaf_tmp.json")
            else:
                logging.info("The temporary JSON file does not exist")

            """
            EVALUATING THE MODEL FOR THE GIVEN SETTING (BOUNDARY DETECTION; SALAMI DATASET)
            """            

            eval.main()

        except Exception as e:
            logging.error(
                f"An error occurred while processing the settings {setting}. Error: {str(e)}. Continuing to the next iteration of the loop.\n{traceback.format_exc()}"
            )


if __name__ == "__main__":
    main()
