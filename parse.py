from master_parser import MasterParser

name = "SALAMI"
min_duration = 3.0
limit = None
base_directory = "/home/oriol_colome_font_epidemicsound_/Master-Thesis/datasets/SALAMI"

parser = MasterParser(
    name=name,
    min_duration=min_duration,
    limit=limit,
    base_directory=base_directory,
)

audio_df, npy_file_name = parser.parse()
