# Dataflow Flex Template PDF to Vertex AI Vector Search
This directory contains code to create, build, and use a Dataflow flex template that:
 1. Ingests PDF files from Google Cloud Storage
 1. Extracts the text from the files 
 1. Chunks the text into smaller chunks
 1. Embeds the chunks
 1. Writes document for each chunk to GCS (to be used in retrieval system for citations and content)
 1. Inserts chunk embedding to Vertex AI Vector Search index 

## Setup 
Run setup.ipynb in a Vertex AI Workbench instance to create and build the flex template.

## Testing
Open and run through example.ipynb to set up infrastructure and test the pipeline on a dataset of research paper PDFs to create a research assistant chatbot. 