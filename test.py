from calmaplab_pipeline import ProcessingStage, CalMAPLabPipeline

pipeline = CalMAPLabPipeline.from_yaml("config.yaml")

# Run only GPS and VOCUS H5 stages
results = pipeline.run(
    "2025-10-08",
    stages=[ProcessingStage.GPS]
)