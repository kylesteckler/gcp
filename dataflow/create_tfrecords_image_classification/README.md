### Beam/Dataflow Pipeline to Create TFRecords for Image Classification
This pipeline prepares image classification data for scalable ML training applications. This pipeline reads image URIs and labels from a BigQuery table, decodes the image, and serializes image/label example to TFRecord format, performs train/test split, and stores the TFRecord files in local storage or GCS. This pipeline can be executed locally or with Cloud Dataflow.

#### Parameters
* `--environment`: str. "local" for running pipeline locally with Apache Beam DirectRunner. "gcp" for running on Cloud Dataflow.
* `--project`: str. Google Cloud project ID.
* `--dataset`: str. Source BigQuery dataset. 
* `--table`: str. Source BigQuery table. 
* `--image_uri_col`: str. BigQuery column name containing the image URIs.
* `--label_col`: str. BigQuery column name containing the labels.
* `--temp_location`: str. GCS location for temporary storage. Required by BigQuery IO connector.
* `--staging_location`: str. GCS location for Dataflow staging. Required only with Dataflow execution. 
* `--output_dir`: str. Output directory to store TFRecord files. Supports local or GCS paths. 
* `--train_percent`: float. Percentage of examples in train set. (E.g. for 80/20 split `--train_percent=0.8`). 
* `--job_name`: str. Job name for Dataflow execution. 
* `--region`: str. Google Cloud region required for Dataflow exeuction. 

#### Local Execution Example
Local execution uses local dependencies and requires the gcloud SDK to be authenticated with your GPC project. 
Local deps needed:
* `apache-beam[gcp]`
* `tensorflow>=2.6` 

Last tested with `apache-beam[gcp]==2.40.0` and `tensorflow==2.8.0`

```
python3 pipeline.py \
    --environment=local \
    --project=${PROJECT} \
    --dataset=${DATASET} \
    --table=${TABLE} \
    --image_uri_col=${IMAGE_URI_COL} \
    --label_col=${LABEL_COL} \
    --temp_location='gs://${BUCKET}/tmp' \
    --output_dir='data/' \
    --train_percent=0.8
```

#### Dataflow Execution Example

```
python3 pipeline.py \
    --environment=gcp \
    --region='us-central1' \
    --project=${PROJECT} \
    --dataset=${DATASET} \
    --table=${TABLE} \
    --image_uri_col=${IMAGE_URI_COL} \
    --label_col=${LABEL_COL} \
    --staging_location='gs://bucket/staging' \
    --temp_location='gs://bucket/tmp' \
    --output_dir='gs://bucket/data' \
    --train_percent=0.8
```
