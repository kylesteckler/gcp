from google.cloud import bigquery 

class Config:
    PROJECT_ID = ""
    DATASET = "covertype_dataset"
    TRAIN_TABLE = "training"
    EVAL_TABLE = "validation"
    
    # Full table schema 
    SCHEMA = [
        bigquery.SchemaField("Elevation", "INTEGER"),
        bigquery.SchemaField("Aspect", "INTEGER"),
        bigquery.SchemaField("Slope", "INTEGER"),
        bigquery.SchemaField("Horizontal_Distance_To_Hydrology", "INTEGER"),
        bigquery.SchemaField("Vertical_Distance_To_Hydrology", "INTEGER"),
        bigquery.SchemaField("Horizontal_Distance_To_Roadways", "INTEGER"),
        bigquery.SchemaField("Hillshade_9am", "INTEGER"),
        bigquery.SchemaField("Hillshade_Noon", "INTEGER"),
        bigquery.SchemaField("Hillshade_3pm", "INTEGER"),
        bigquery.SchemaField("Horizontal_Distance_To_Fire_Points", "INTEGER"),
        bigquery.SchemaField("Wilderness_Area", "STRING"),
        bigquery.SchemaField("Soil_Type", "STRING"),
        bigquery.SchemaField("Cover_Type", "INTEGER"),
    ]
    
    # Columns to exclude from ingest
    COLS_TO_EXCLUDE = [] 
    
    # SQL text filtering statement
    # similar to a WHERE clause in a query.
    ROW_RESTRICTION = None 
    
    # Desirable number of streams that can be read in parallel
    NUM_STREAMS = 1
    
    # Numeric columns to ingest. INTEGER or FLOAT 
    NUMERIC_COLS = [
        "Elevation",
        "Aspect",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Hillshade_9am",
        "Hillshade_Noon",
        "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points",
    ]
    
    # Categorical columns to ingest. STRING
    CATEGORICAL_COLS = [
        "Wilderness_Area",
        "Soil_Type"
    ]
    
    CATEGORICAL_VOCAB = {
        "Wilderness_Area": [
            "Cache", "Commanche", "Rawah", "Neota"
        ],
        "Soil_Type": [
            'C2702','C2703','C2705','C2717','C3501','C3502',
            'C4201','C4703','C4704','C4744','C4758','C5101',
            'C6101','C6102','C6731','C7101','C7102','C7103',
            'C7201','C7202','C7700','C7702','C7745','C7746',
            'C7755','C7756','C7757','C8703','C8771','C8772',
            'C8776','C2706','C7709','C7710','C7790','C2704',
            'C8708','C5151','C7701','C8707'
        ]
    }
    
    # Label column
    LABEL_COL = "Cover_Type"
    
    # Number of label classes 
    NUM_CLASSES = 7
