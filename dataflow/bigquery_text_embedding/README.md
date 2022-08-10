### Beam/Dataflow pipeline for generating embeddings for a text field in a BigQuery table. 

This pipeline uses Google's [Universal Sentence Encoder](https://tfhub.dev/google/universal-sentence-encoder/4) to generate 512 dimensional embeddings from a text input. If you wish to use another model you need to change the url in the `EmbedDoFn`'s setup method. The pipeline creates a new BigQuery table that is identical to the source table, with an additional array column: the embedding of the text column specified. 

#### Parameters
- `project`: Google Cloud project ID. 
- `region`: Google Cloud region. Default is `us-central1`.
- `dataset`: BigQuery dataset of source/sink tables.
- `input_table`: BigQuery table name of source table.
- `output_table`: BigQuery table name of sink table.
- `text_col`: The column name in the source table with text data to embed. 
- `staging_location`: GCS path for staging.
- `temp_location`: GCS path for temporary storage.
- `requirements_file`: File for Python deps.

#### Example Usage
```
python3 pipeline.py \
   --project=${PROJECT_ID} \
   --region=${REGION} \
   --dataset=${BIGQUERY_DATASET} \
   --input_table=${SOURCE_TABLE_NAME} \
   --output_table=${SINK_TABLE_NAME} \
   --text_col=${COLUMN_TO_EMBED} \
   --staging_location=gs://${BUCKET}/staging \
   --temp_location=gs://${BUCKET}/tmp \
   --requirements_file='./requirements.txt'
```
