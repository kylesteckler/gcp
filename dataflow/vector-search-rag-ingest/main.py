import logging
import json
import apache_beam as beam 
from apache_beam.options.pipeline_options import (
    PipelineOptions, 
    StandardOptions, 
    GoogleCloudOptions, 
    SetupOptions
)
import argparse 
import uuid
import tensorflow as tf
import tensorflow_hub as hub
import gcsfs
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.cloud import storage 
from google.cloud import aiplatform 
from google.cloud.aiplatform_v1.types.index import IndexDatapoint
from typing import List, Optional


def extract_pdf_text(file_path: str, project_id: str):
    """
    Reads PDF file from GCS.
    Returns the pages of extracted text, page numbers, and source file.
    """
    try: 
        file_system = gcsfs.GCSFileSystem(project=project_id)
        file_object = file_system.open(file_path, "rb")
        reader = PdfReader(file_object, strict=False)
        pages = [
            {
                "file_path": file_path,
                "content": page.extract_text(),
                "page_number": i+1
            }
            for i, page in enumerate(reader.pages)
        ]
        return pages
    except:
        # todo: error handling and logging here when pypdf fails at end of stream
        return []

        
def chunk_text(element: dict, chunk_size: int, chunk_overlap: int):
    """
    Chunks text into windows of chunk_size.
    Scrolls through text with stride of chunk_overlap. 
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    texts = text_splitter.create_documents([element["content"]])
    return [
        {
            "content": doc.page_content,
            "page_number": element["page_number"],
            "source_file": element["file_path"]
        }
        for doc in texts    
    ]


class TextEmbeddingDoFn(beam.DoFn):
    """
    PTransform to use Google's universal sentence encoder for text embeddings.
    
    Alternatively you could use a paid API here (e.g. Vertex AI Text Embeddings) instead of 
    loading an OSS embedding model in memory. 
    """
    def __init__(self, module_url: str = "https://tfhub.dev/google/universal-sentence-encoder/4"):
        self.module_url = module_url 
        
    def setup(self):
        self.embedding_model = hub.load(self.module_url)
        
    def process(self, element):
        embedding = self.embedding_model([element["content"]])

        yield {
            "content": element["content"],
            "page_number": element["page_number"],
            "source_file": element["source_file"],
            "embedding": tf.squeeze(embedding).numpy().tolist(),
            "id": uuid.uuid4().hex
        }
        
        
class WriteChunkDocumentDoFn(beam.DoFn):
    """
    PTransform that writes a JSON file to GCS with the chunk content and metadata.
    Name of the file is documents/{embedding_id}.json
    """
    def __init__(self, bucket: str):
        self.bucket_name = bucket
    
    def setup(self):
        self.bucket = storage.Client().get_bucket(self.bucket_name) 
    
    def process(self, element):
        blob = self.bucket.blob(f'documents/{element["id"]}.json')
        doc_content = {
            "content": element["content"],
            "page_number": element["page_number"],
            "source_file": element["source_file"]
        }
        blob.upload_from_string(
            data=json.dumps(doc_content),
            content_type='application/json'
        )
        yield element 

            
class WriteChunkVectorDoFn(beam.DoFn):
    """
    PTransform that inserts vector to Vertex AI Vector Search index 
    """
    def __init__(self, index_id: str):
        self.index_id = index_id 
        
    def setup(self):
        self.index = aiplatform.MatchingEngineIndex(self.index_id) 
    
    def process(self, element):
        datapoint = IndexDatapoint(
            datapoint_id=element["id"],
            feature_vector=element["embedding"]
        )
        result = self.index.upsert_datapoints([datapoint])

def run(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_id", required=True, help="Google Cloud Project ID")
    parser.add_argument("--dataset_file", required=True, help="GCS path to file with list of PDF paths")
    parser.add_argument("--index_id", required=True, help="Vertex AI Vector Search Index ID")
    parser.add_argument("--document_bucket", required=True, help="Bucket name to store chunk documents")
    parser.add_argument("--chunk_size", required=True, help="Size of text chunks computed by len(characters)")
    parser.add_argument("--chunk_overlap", required=True, help="Number of characters to overlap each chunk")
    
    pipeline_args, pipeline_options_args = parser.parse_known_args(argv)
    
    project_id = pipeline_args.project_id
    dataset_file = pipeline_args.dataset_file
    index_id = pipeline_args.index_id
    document_bucket = pipeline_args.document_bucket
    chunk_size = int(pipeline_args.chunk_size)
    chunk_overlap = int(pipeline_args.chunk_overlap)
    
    # Needed for dataflow runner 
    pipeline_options_args.append(f"--project={project_id}")
    pipeline_options_args.append("--pickle_library=cloudpickle")
    
    pipeline_options = PipelineOptions(pipeline_options_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    
    with beam.Pipeline(options=pipeline_options) as p:
        _ = (
            p
            | "Read Dataset File" >> beam.io.ReadFromText(dataset_file)
            | "Extract PDF Text" >> beam.FlatMap(extract_pdf_text, project_id=project_id)
            | "Chunk Text" >> beam.FlatMap(chunk_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            | "Compute Embedding" >> beam.ParDo(TextEmbeddingDoFn())
            | "Write Document" >> beam.ParDo(WriteChunkDocumentDoFn(document_bucket))
            | "Insert Vector" >> beam.ParDo(WriteChunkVectorDoFn(index_id))
        )
    
    p.run()
    
    
if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()