import tensorflow as tf 
import tensorflow_io 
import fire 
from typing import List 
import os 

from tensorflow.python.framework import dtypes
from tensorflow_io.bigquery import BigQueryClient
from tensorflow_io.bigquery import BigQueryReadSession

from config import Config

def parse_row(row):
    # Create dictionary of column_name: column_value
    # Remove spaces on start/end of all string features 
    features = { name: 
                (tf.strings.strip(val) if val.dtype == 'string' \
                 else tf.cast(val, tf.float32))
                for (name, val) in row.items()
              }
    
    # Assumes categorical label is a positive integer
    label = features.pop(Config.LABEL_COL)
    
    return (features, label)

def create_dataset(batch_size, mode="train"):
    if mode=="train":
        table = Config.TRAIN_TABLE
    else:
        table = Config.EVAL_TABLE
        
    output_types = []
    selected_fields = [] 
    for field in Config.SCHEMA:
        if not field.name in Config.COLS_TO_EXCLUDE:
            selected_fields.append(field.name)
            if field.field_type == 'INTEGER':
                output_types.append(dtypes.int64)
            elif field.field_type == 'FLOAT' or field.field_type == 'FLOAT64':
                output_types.append(dtypes.double)
            else:
                output_types.append(dtypes.string)
        
    tfio_bigquery_client = BigQueryClient()
        
    read_session = tfio_bigquery_client.read_session(
        parent=f"projects/{Config.PROJECT_ID}",
        project_id=Config.PROJECT_ID,
        dataset_id=Config.DATASET,
        table_id=table,
        selected_fields=selected_fields,
        output_types=output_types,
        row_restriction=Config.ROW_RESTRICTION,
        requested_streams=Config.NUM_STREAMS
    )

    ds = read_session.parallel_read_rows(
        sloppy=True,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds = ds.map(parse_row)
    
    # Repeat indefinitely for training
    if mode == "train":
        return ds.batch(batch_size).repeat() 
    
    # Repeat through validation set once 
    return ds.batch(batch_size).repeat(1)

def create_input_layers():
    """
    Returns dictionary of `tf.Keras.layers.Input` layers for each feature.
    """
    num_inputs = {
        col: tf.keras.layers.Input(
            name=col, shape=(1,), dtype="float32"
        )
        for col in Config.NUMERIC_COLS
    }

    cat_inputs = {
        col: tf.keras.layers.Input(
            name=col, shape=(1,), dtype="string")
        for col in Config.CATEGORICAL_COLS
    }

    inputs = {**num_inputs, **cat_inputs}

    return inputs

def transform(inputs):
    """
    Define feature transforms with Keras preprocessing layers
    """
    
    transformed = {} 
    
    # Pass along numeric cols as is 
    for col in Config.NUMERIC_COLS:
        transformed[col] = inputs[col]
    
    # One hot encode categorical cols 
    for col in Config.CATEGORICAL_COLS:
        transformed[col] = tf.keras.layers.StringLookup(
            vocabulary=Config.CATEGORICAL_VOCAB[col], 
            output_mode="one_hot"
        )(inputs[col])
        
    return transformed 

def build_model(dnn_hidden_units: List[int]):
    inputs = create_input_layers()
    transformed = transform(inputs)
    model_inputs = tf.keras.layers.Concatenate()(
        transformed.values())

    # First hidden layer 
    x = tf.keras.layers.Dense(
        units=dnn_hidden_units[0], activation="relu", name="h1")(model_inputs)
    
    for i in range(1, len(dnn_hidden_units)):
        x = tf.keras.layers.Dense(
            units=dnn_hidden_units[i], activation="relu", name=f"h{i+1}")(x)
    
    output = tf.keras.layers.Dense(
        units=Config.NUM_CLASSES, activation="softmax", 
        name="output")(x)
    
    return tf.keras.models.Model(inputs=inputs, outputs=output)

def train_evaluate(
    output_dir: str ='gs://kylesteckler-instructor/testing/model',
    batch_size: int = 32,
    num_train_examples: int = 1000000,
    num_evals: int = 10,
    dnn_hidden_units: List[int] = [128, 128]
):
    
    steps_per_epoch = num_train_examples // (batch_size * num_evals)
    train_ds = create_dataset(batch_size=batch_size, mode="train")
    val_ds = create_dataset(batch_size=batch_size, mode="eval")
    
    model = build_model(dnn_hidden_units)
    model.compile(optimizer='adam', metrics=['accuracy'],
                loss=tf.keras.losses.SparseCategoricalCrossentropy())
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_evals,
        steps_per_epoch=steps_per_epoch
    )
    
    model.save(output_dir) 
    
if __name__ == '__main__':
    fire.Fire(train_evaluate)
