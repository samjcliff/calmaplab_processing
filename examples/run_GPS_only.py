from calmaplab_pipeline import ProcessingStage, CalMAPLabPipeline

pipeline = CalMAPLabPipeline.from_yaml("config.yaml")

# Run only GPS and VOCUS H5 stages
results = pipeline.run(
    "2026-02-03",
    stages=[ProcessingStage.GPS],
)