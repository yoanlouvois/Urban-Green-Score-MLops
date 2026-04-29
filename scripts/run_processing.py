import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker import image_uris
import os
from dotenv import load_dotenv

load_dotenv()

ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")
BUCKET = os.getenv("S3_BUCKET")
REGION = os.getenv("AWS_REGION")


def main():
    session = sagemaker.Session()

    image_uri = image_uris.retrieve(
        framework="pytorch",
        region=REGION,
        version="2.2.0",
        py_version="py310",
        instance_type="ml.m5.large",
        image_scope="training",
    )

    processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        role=ROLE_ARN,
        instance_count=1,
        instance_type="ml.m5.large",
        base_job_name="urban-green-preprocess",
        sagemaker_session=session,
    )

    processor.run(
        code="src/preprocessing/preprocess.py",
        source_dir="src",
        inputs=[
            ProcessingInput(
                source=f"s3://{BUCKET}/data/raw",
                destination="/opt/ml/processing/input/raw",
            )
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output/processed",
                destination=f"s3://{BUCKET}/data/processed",
            )
        ],
        arguments=[
            "--raw-data-dir", "/opt/ml/processing/input/raw",
            "--output-dir", "/opt/ml/processing/output/processed",
            "--num-workers", "4",
        ],
    )


if __name__ == "__main__":
    main()