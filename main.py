import itertools
import os
import logging
import traceback
import time
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
        "MAX_EPOCHS": 1,
        "EVAL_WINDOW": 0.5,
    }

    batch_sizes = sorted([4, 5, 6, 7, 8])
    window_sizes = sorted([i * settings_dict["SAMPLE_RATE"] for i in range(4, 9)])
    feature_list = sorted(["pcp", "mfcc", "embeddiogram"])
    max_epochs_list = sorted([10, 50, 100, 200, 300, 400, 500, 1000])
    clip_durations = sorted([7.0, 11.0, 15.0])

    settings = []

    for (
        batch_size,
        window_size,
        feature,
        max_epochs,
        clip_duration,
    ) in itertools.product(
        batch_sizes, window_sizes, feature_list, max_epochs_list, clip_durations
    ):
        settings_item = settings_dict.copy()
        settings_item.update(
            {
                "BATCH_SIZE": batch_size,
                "WINDOW_SIZE": window_size,
                "FEATURE": feature,
                "MAX_EPOCHS": max_epochs,
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

            # Create a file to signal that training is done
            with open('training_done.txt', 'w') as f:
                f.write('Training completed')

            # Wait for the training to complete
            while not os.path.exists('training_done.txt'):
                time.sleep(1)

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

            # Remove the file to signal the end of evaluation and ready for next training
            os.remove('training_done.txt')
        except Exception as e:
            logging.error(
                f"An error occurred while processing the settings {setting}. Error: {str(e)}. Continuing to the next iteration of the loop.\n{traceback.format_exc()}"
            )


if __name__ == "__main__":
    main()
