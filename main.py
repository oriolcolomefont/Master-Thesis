import itertools
import os
import train
import features
from msaf import config
import eval
from tqdm import tqdm


settings_dict = {
    "FILE_LIST_PATH": "./datasets/MSD/MSD_audio_limit=all.csv",
    "DATASET_NAME": "Million Song Dataset",
    "SAMPLE_RATE": config.sample_rate,
    "MAX_EPOCHS": 1,
    "EVAL_WINDOW": 0.5,
}

batch_sizes = [4, 5, 6, 7, 8]
window_sizes = [i * settings_dict["SAMPLE_RATE"] for i in range(4, 9)]
feature_list = ["pcp", "mfcc", "embeddiogram"]
max_epochs_list = [10, 50, 100, 200, 300, 400, 500, 1000]

settings = []

for batch_size, window_size, feature, max_epochs in itertools.product(
    batch_sizes, window_sizes, feature_list, max_epochs_list
):
    settings_item = settings_dict.copy()
    settings_item.update(
        {
            "BATCH_SIZE": batch_size,
            "WINDOW_SIZE": window_size,
            "FEATURE": feature,
            "MAX_EPOCHS": max_epochs,
        }
    )
    settings.append(settings_item)


for setting in tqdm(settings):
    try:
        train.FILE_LIST_PATH = setting["FILE_LIST_PATH"]
        train.DATASET_NAME = setting["DATASET_NAME"]
        train.BATCH_SIZE = setting["BATCH_SIZE"]
        train.CLIP_DURATION = setting["CLIP_DURATION"]
        train.SAMPLE_RATE = setting["SAMPLE_RATE"]
        train.MAX_EPOCHS = setting["MAX_EPOCHS"]

        fit, best_model_path = train.main()
        eval.FEATURE = setting["FEATURE"]
        eval.EVAL_WINDOW = setting["EVAL_WINDOW"]
        features.CKPT_PATH = best_model_path
        features.WINDOW_SIZE = setting["WINDOW_SIZE"]

        # Check if a file exists before attempting to delete it
        if os.path.exists("/home/jupyter/oriol/Master-Thesis/.features_msaf_tmp.json"):
            os.remove("/home/jupyter/oriol/Master-Thesis/.features_msaf_tmp.json")
        else:
            print("The temporary JSON file does not exist")
        eval.main()
    except Exception as e:
        print(
            f"An error occurred while processing the settings {setting}. Error: {str(e)}. Continuing to the next iteration of the loop."
        )
