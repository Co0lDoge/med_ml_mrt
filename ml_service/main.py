from concurrent import futures

import grpc
from minio import Minio

import ml_service.internal.events.events as kafkaevents
import ml_service.internal.ml_model.classification_efficientnet as cla
import ml_service.internal.ml_model.segmentation as seg
import ml_service.internal.ml_model.kt.Classification_ResNet as ktcla
import ml_service.internal.ml_model.kt.Segmentation_YOLO as ktseg
import ml_service.internal.s3.s3 as mys3
import ml_service.internal.usecases.mri.mri as usecasemri
import ml_service.internal.usecases.kt.kt as usecasekt
from ml_service.config.default import get_settings


def run_server():
    settings = get_settings()

    minio_client = Minio(
        endpoint=settings.s3_endpoint,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key.get_secret_value(),
        secure=False,  # Установите True, если используете HTTPS
    )

    mri_bucket = settings.s3_mri_bucket_name
    kt_bucket = settings.s3_kt_bucket_name

    mri_s3 = mys3.S3(minio_client, mri_bucket)
    kt_s3 = mys3.S3(minio_client, kt_bucket)

    if settings.segmentation_model_type == "nnunet":
        import ml_service.internal.ml_model.nnunet_segmentation as nn_seg
        segmdl = nn_seg.NnUNetSegmentationModel(model_dir=settings.nnunet_model_dir)
    else:
        segmdl = seg.SegmentationModel(model_type=settings.segmentation_model_type)
        
    claml = cla.EfficientNetModel(model_type=settings.classification_model_type)
    ktcla_model = ktcla.ClassificationModel()
    ktseg_model = ktseg.SegmentationModel()

    usecaseMri = usecasemri.mriUseCase(segmdl, claml, mri_s3)
    usecaseKt = usecasekt.ktUseCase(ktseg_model, ktcla_model, kt_s3)

    kafka = kafkaevents.EventsYo(usecaseMri, usecaseKt)
    print("Kafka started...")
    kafka.run()


if __name__ == "__main__":
    run_server()
