import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions, GoogleCloudOptions, SetupOptions
from google.cloud import bigquery 
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

class EmbedDoFn(beam.DoFn):
    
    # called whenever DoFn instance is deserialized on the worker
    def setup(self):
        self.model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        
    def process(self, element, text_col):
        embedding = self.model([element[text_col]]) # tf.Tensor(shape (1, num_emb))
        embedding_col_name = f'{text_col}_embedding'
        element[embedding_col_name] = tf.squeeze(
            embedding).numpy().tolist()
        
        yield element
        
def run():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--project',
        type=str,
        help='Google Cloud Project'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        default='us-central1',
        help='Google Cloud region. Required for Dataflow execution'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='BigQuery Dataset ID that contains source table'
    )

    parser.add_argument(
        '--input_table',
        type=str,
        help='BigQuery Table ID for input table'
    )
    
    parser.add_argument(
        '--output_table',
        type=str,
        help='BigQuery Table ID for table to be created.'
    )

    parser.add_argument(
        '--text_col',
        type=str,
        help='Name of the column to generate text embeddings for.'
    )
    
    parser.add_argument(
        '--staging_location',
        type=str,
        help='Staging GCS location'
    )
    
    parser.add_argument(
        '--temp_location',
        type=str,
        help='Temp location for BigQuery IO connector' 
    )
    
    parser.add_argument(
        '--job_name',
        type=str,
        default='bq-text-embedding',
        help='Job name for dataflow execution'
    )
        
    args, beam_args = parser.parse_known_args()
    
    pipeline_options = PipelineOptions(beam_args)
        
    # Make global imports available to all workers (if distributed execution)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    google_cloud_options = pipeline_options.view_as(GoogleCloudOptions)
    
    # Set required GCP options
    google_cloud_options.project = args.project
    google_cloud_options.temp_location = args.temp_location
    
    pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'
    google_cloud_options.region = args.region
    google_cloud_options.job_name = args.job_name
    google_cloud_options.staging_location = args.staging_location 
    
    # For beam io
    table_spec = f'{args.project}:{args.dataset}.{args.input_table}'
    output_table_spec = f'{args.project}:{args.dataset}.{args.output_table}'

    # For bq client to get schema 
    table_ref = f'{args.project}.{args.dataset}.{args.input_table}'
    
    client = bigquery.Client()
    table = client.get_table(table_ref)
    input_schema = table.schema 
    
    # Add embedding column to input table schema for output schema
    embedding_col = [bigquery.SchemaField(
        f'{args.text_col}_embedding', 'FLOAT', 'REPEATED')]
    
    output_schema = input_schema + embedding_col
    
    # Convert from bigquery.SchemaField list to dict for beam io
    output_schema_dict = {
        'fields': [
            {'name': x.name, 'type': x.field_type, 'mode': x.mode} for x in output_schema
        ]
    }
    
    
    with beam.Pipeline(options=pipeline_options) as p:
        _ = ( p
             | 'Read Table' >> beam.io.ReadFromBigQuery(
                 table=table_spec)
             | 'Generate Embeddings' >> beam.ParDo(
                 EmbedDoFn(), text_col=args.text_col)
             | 'Write BigQuery' >> beam.io.WriteToBigQuery(
                 output_table_spec,
                 schema=output_schema_dict,
                 write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                 create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                 method="FILE_LOADS"                 
             )
            )
        
    p.run()

if __name__ == '__main__':
    run()
