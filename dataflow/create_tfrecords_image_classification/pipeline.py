import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions, GoogleCloudOptions, SetupOptions
from apache_beam.io.gcp.internal.clients import bigquery
import argparse
import typing
import tensorflow as tf 
import random
import os 

# DoFn to create TF Example's
class CreateTFExample(beam.DoFn):
    def _image_feature(self, value):
        """Returns a bytes_list from an image."""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
        )

    def _bytes_feature(self, value):
        """Returns bytes_list from a string feature"""
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value])
        )
    
    def process(self, element):
        img = tf.io.decode_jpeg(tf.io.read_file(element["image_uri"]))
        
        feature = {
            "image": self._image_feature(img), 
            "label": self._bytes_feature(element["label"].encode()),
        }
        
        yield tf.train.Example(features=tf.train.Features(feature=feature))
    
class ParseCols(beam.DoFn):
    def process(self, element, image_uri_col, label_col):
        yield {
            "image_uri": element[image_uri_col],
            "label": element[label_col]
        }
        
# random train/test split
def partition_fn(example, num_partitions, train_percent):
    if random.random() < train_percent:
        return 0
    return 1

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--environment',
        choices=['local', 'gcp'],
        help='local to run with DirectRunner. gcp for DataflowRunner'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        help='Google Cloud Project'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='BigQuery Dataset ID that contains source table'
    )

    parser.add_argument(
        '--table',
        type=str,
        help='BigQuery Table ID. Must have columns for image URI and label'
    )

    parser.add_argument(
        '--image_uri_col',
        type=str,
        help='Name of image URIs column in BigQuery table'
    )

    parser.add_argument(
        '--label_col',
        type=str,
        help='Name of label column in BigQuery table'
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
        '--output_dir',
        type=str,
        help='Output directory for TFRecords'
    )
    
    parser.add_argument(
        '--train_percent',
        type=float,
        help='Percent of data (0-1) desired in training split'
    )
    
    parser.add_argument(
        '--region',
        type=str,
        help='Google Cloud region. Required for Dataflow execution'
    )
    
    parser.add_argument(
        '--job_name',
        type=str,
        default='bq-to-tfrecords-classification',
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


    if args.environment == 'gcp':
        assert args.region, "You must specify a Google Cloud region to submit a Dataflow job"
        assert args.staging_location, "You must specify a GCS staging location for Dataflow jobs"
        
        pipeline_options.view_as(StandardOptions).runner = 'DataflowRunner'
        google_cloud_options.region = args.region
        google_cloud_options.job_name = args.job_name
        google_cloud_options.staging_location = args.staging_location 


    if args.environment == 'local':
        pipeline_options.view_as(StandardOptions).runner = 'DirectRunner'
        
    # Source table 
    table_spec = bigquery.TableReference(
        projectId=args.project,
        datasetId=args.dataset,
        tableId=args.table
    )
    
    with beam.Pipeline(options=pipeline_options) as p:

        train, val = ( p
                    | 'Read Table' >> beam.io.ReadFromBigQuery(
                        table=table_spec)
                    | 'Parse Columns' >> beam.ParDo(
                        ParseCols(), 
                        image_uri_col=args.image_uri_col, 
                        label_col=args.label_col)
                    | 'Create TF Examples' >> beam.ParDo(CreateTFExample())
                    | 'Serialize Examples' >> beam.Map(
                        lambda x: x.SerializeToString()
                    )
                    | 'Split Data' >> beam.Partition(
                        partition_fn, 2, train_percent=args.train_percent
                    )
                   )
        
        _ = train | 'Write Train' >> beam.io.tfrecordio.WriteToTFRecord(
            os.path.join(args.output_dir, 'train.tfrecord')
        )
        
        _ = val | 'Write Validation' >> beam.io.tfrecordio.WriteToTFRecord(
            os.path.join(args.output_dir, 'val.tfrecord')
        )

    p.run()
    
if __name__ == '__main__':
    run()
