from msd_parser import MSDParser

msd_parser = MSDParser(clip_duration=8.0, sample_rate=41000, limit=10000)
audio_df = msd_parser.parse()
