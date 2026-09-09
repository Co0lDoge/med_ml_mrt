
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class DefaultSettings(BaseSettings):
    s3_endpoint: str = "localhost:9000"
    s3_access_key: str = ""
    s3_secret_key: SecretStr
    s3_mri_bucket_name: str = "mri"
    s3_kt_bucket_name: str = "kt-b"

    grpc_port: int = 50055

    kafka_host: str = "localhost"
    kafka_port: int = 9092

    segmentation_model_type: str = "nnunet"
    classification_model_type: str = "cross"
    nnunet_model_dir: str = "./models/nnunet"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def get_settings() -> DefaultSettings:
    return DefaultSettings()
