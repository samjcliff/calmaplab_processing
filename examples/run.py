from calmaplab_pipeline import CalMAPLabPipeline

# One-liner from config file
pipeline = CalMAPLabPipeline.from_yaml("config.yaml")
results = pipeline.run("2026-01-30")