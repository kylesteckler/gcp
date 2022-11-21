import tensorflow as tf
import collections 
from typing import Dict, Sequence, Any, Tuple, Union
from tensorflow_io.bigquery import BigQueryClient
from google.cloud import bigquery 


def create_train_eval_datasets(
    project_id: str,
    dataset: str,
    table: str,
    target_col: str,
    target_vocab: Sequence[Union[str, int]] = None,
    fields_to_exclude: Sequence[str] = None,
    eval_table: str = None,
    eval_split: float = 0.2,
    batch_size: int = 64,
    eval_batch_size: int = 64,
    epochs: int = None,
    requested_streams: int = 1,
    shuffle_buffer: int = None,
    eval_shuffle_buffer: int = None,
    return_keras_input_layers: bool = False
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Creates a training and validation tf.data.Dataset from a BigQuery source.
    
    Inputs:
     - project_id: Required. Google Cloud project ID. 
     - dataset: Required. BigQuery dataset.
     - table: BigQuery table of training data.
     - target_col: Name of the label column.
     - target_vocab: Optional. A list of label values. If provided, each label value will be mapped to 
                     its index in target_vocab. 
     - fields_to_exclude: Optional. A list of columns in the table to exclude. 
     - eval_table: Optional. A seperate BigQuery table in the same dataset to use as validation data.
     - eval_split: Optional (default=0.2). Fraction of data to be used for validation if eval_table is not provided.
     - batch_size: Optional (default=64). Training batch size.
     - eval_batch_size. Optional (default=64). Validation batch size.
     - epochs: Optional (default=None). Number of times to iterate through the dataset. If epochs=None it will 
               iterate through the dataset indefinitely.
     - requested_streams: Optional (default=1). Number of requested streams in the BigQuery read session.
     - shuffle_buffer: Optional. Number of rows to read into the shuffle buffer of the training dataset.
     - eval_shuffle_buffer: Optional. Number of rows to read into the shuffle buffer of the eval dataset.
     - return_keras_input_layers: Optional (default=False). If True, function returns Tuple of 
                                 (train_dataset, eval_dataset, keras_input_layers) where keras_input_layers
                                 is a list of tf.keras.layers.Input layers with names and types of each 
                                 feature in the training dataset. These input layers can be used directly to 
                                 create and train a tf.keras model on the dataset returned. 
     
    Returns:
     - Tuple of the training and validation datasets. Tuple[tf.data.Dataset, tf.data.Dataset]. Each element in 
       both datasets is a Tuple[features, label] where features is a dictionary of Tensors and label is a Tensor, 
       compatible with the .fit method of tf.keras models. 
    
     - Optional (if return_keras_input_layers):
                Tuple[tf.data.Dataset, tf.data.Dataset, List[tf.keras.layers.Input]]
    
    """
    
    schema = _get_bigquery_schema(project_id, dataset, table)
    selected_fields = _get_selected_fields(schema, fields_to_exclude)
    
    if eval_table:
        train_dataset = _create_dataset(
            project_id=project_id,
            dataset=dataset,
            table=table,
            selected_fields=selected_fields,
            farm_fingerprint=False,
            requested_streams=requested_streams
        )
        
        eval_dataset = _create_dataset(
            project_id=project_id,
            dataset=dataset,
            table=eval_table,
            selected_fields=selected_fields,
            farm_fingerprint=False,
            requested_streams=requested_streams
        )
    
    else:
        dataset = _create_dataset(
            project_id=project_id,
            dataset=dataset,
            table=table,
            selected_fields=selected_fields,
            farm_fingerprint=True,
            requested_streams=requested_streams
        )
        
        train_dataset, eval_dataset = _split_dataset(dataset, eval_split)
        
    if not shuffle_buffer:
        shuffle_buffer = 10 * batch_size 
        
    if not eval_shuffle_buffer:
        eval_shuffle_buffer = 10 * eval_batch_size 
        
    train_dataset = _preprocess_dataset(
        dataset=train_dataset,
        target_col=target_col,
        batch_size=batch_size,
        repeat=epochs,
        shuffle_buffer_size=shuffle_buffer,
        target_vocab=target_vocab
    )
    
    eval_dataset = _preprocess_dataset(
        dataset=eval_dataset,
        target_col=target_col,
        batch_size=eval_batch_size,
        repeat=1,
        shuffle_buffer_size=eval_shuffle_buffer,
        target_vocab=target_vocab
    )
    
    if return_keras_input_layers:
        input_layers = _get_keras_input_layers(selected_fields, target_col) 
        return (train_dataset, eval_dataset, input_layers)
    
    return train_dataset, eval_dataset 
        

def _preprocess_dataset(
    dataset: tf.data.Dataset,
    target_col: str,
    batch_size: int,
    repeat: int,
    shuffle_buffer_size: int,
    target_vocab: Sequence[Union[str, int]]) -> tf.data.Dataset:
    
    ds = dataset.map(
        lambda x: _parse_feature_dict(x, target_col=target_col, target_vocab=target_vocab),
        num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.shuffle(shuffle_buffer_size).batch(batch_size).repeat(repeat)
    
    return ds.prefetch(tf.data.AUTOTUNE) 


def _parse_feature_dict(
    x: collections.OrderedDict,
    target_col: str,
    target_vocab: Sequence[Union[str, int]] = None
) -> Tuple[collections.OrderedDict, tf.Tensor]:
    
    # Ordered dict of all fields 
    features = x
    
    target = features.pop(target_col)
    
    # Map categorical label to integer index in vocab list if needed
    if target_vocab:
        label = _map_label_to_int(target, target_vocab)
    
    return features, label


@tf.function 
def _map_label_to_int(
    label: tf.Tensor, 
    vocab: Sequence[Union[str,int]]) -> tf.Tensor:
    """Takes in a categorical label and returns the integer index of that label in a vocabulary list"""
    pred_fn_pairs = []
        
    for i in range(len(vocab)):
        pred_fn_pairs.append(
            (tf.equal(label, vocab[i]), lambda: i)
        )
        
    return tf.case(pred_fn_pairs)


def _create_dataset(
    project_id: str,
    dataset: str,
    table: str,
    selected_fields: Dict[str, Dict[str, Any]],
    farm_fingerprint: bool,
    requested_streams: int) -> tf.data.Dataset:
    
    # Tensorflow wrapper around BQ Storage Read API
    tfio_bigquery_client = BigQueryClient()

    read_session = tfio_bigquery_client.read_session(
        parent=f"projects/{project_id}",
        project_id=project_id,
        dataset_id=dataset,
        table_id=table,
        selected_fields=selected_fields,
        requested_streams=requested_streams
    )
    
    # sloppy as we dont need deterministic order of rows
    ds = read_session.parallel_read_rows(
        sloppy=True,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    if farm_fingerprint:
        # Calculate fingerprint to split train and eval datasets
        ds = ds.map(
            _get_fingerprint,
            num_parallel_calls=tf.data.AUTOTUNE)
    
    return ds 


def _split_dataset(
    ds: tf.data.Dataset,
    eval_split: float) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    
    eval_percent = round(eval_split * 100)
    if eval_percent < 0 or eval_percent > 100:
        raise ValueError('Eval split must be between [0,1]')
    
    # Mod on the hash to seperate splits
    train_ds = ds.filter(lambda fp, feature_dict: fp % 100 > eval_percent)
    eval_ds = ds.filter(lambda fp, feature_dict: fp % 100 <= eval_percent)
    
    # Get rid of the fingerprint 
    train_ds = train_ds.map(lambda fp, feature_dict: feature_dict)
    eval_ds = eval_ds.map(lambda fp, feature_dict: feature_dict) 
    
    return train_ds, eval_ds
    

def _get_fingerprint(x: collections.OrderedDict) -> Tuple[tf.Tensor, collections.OrderedDict]:
    """Returns farmhash64 (fingerprint) of feature dict and feature dict itself """
    tensors = [tf.expand_dims(t,0) for t in list(x.values())]
    fps = [tf.fingerprint(t) for t in tensors]
    fps = tf.cast(fps, tf.int32)
    fp = tf.reduce_sum(fps)
    return fp, x
    
    
def _get_selected_fields(
    schema: Dict[str, bigquery.SchemaField],
    fields_to_exclude: Sequence[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Creates selected fields for BigQuery read session. This is a dictionary with key as column name and
        value as a dictionary with key output_type and value tensorflow type. 
    """
    BIGQUERY_TF_MAP = {
        'FLOAT': tf.float64,
        'FLOAT64': tf.float64,
        'INTEGER': tf.int64,
        'STRING': tf.string
    }
    
    if fields_to_exclude:
        missing_cols = set(fields_to_exclude).difference(schema)
        if missing_cols:
            raise ValueError(f'Fields to exclude: {missing_cols} not found in table.')
            
        # Remove fields to exclude 
        schema = {k:v for k,v in schema.items() if k not in fields_to_exclude}
    
    selected_fields = {}
    for field_name, field in schema.items():
        bq_type = field.field_type
        
        if bq_type not in BIGQUERY_TF_MAP:
            raise ValueError(
                f'Column: {field_name} has unsupported type: {bq_type}. '
                f'Supported BigQuery types: {list(BIGQUERY_TF_MAP)} '
            )
        
        selected_fields[field_name] = {'output_type': BIGQUERY_TF_MAP[bq_type]}
    
    return selected_fields 


def _get_bigquery_schema(
    project_id: str,
    dataset: str,
    table: str
) -> Dict[str, bigquery.SchemaField]:
    """Use BigQuery API to return dictionary of column name: schema for all columns in specified BQ table."""
    
    client = bigquery.Client() 
    dataset_ref = client.dataset(dataset, project=project_id)
        
    table_ref = dataset_ref.table(table)
    table = client.get_table(table_ref)
    
    schema = table.schema
    schema_dict = {col.name: col for col in schema}
    return schema_dict 


def _get_keras_input_layers(
    selected_fields: Dict[str, Dict[str, Any]],
    target_col: str) -> Sequence[tf.keras.layers.Input]:
    
    # Remove label column 
    _ = selected_fields.pop(target_col)
    
    keras_input_layers = list()
    for feature_name, feature_config in selected_fields.items():
        keras_input_layers.append(
            tf.keras.layers.Input(
                name=feature_name,
                shape=(1,),
                dtype=feature_config['output_type']
            )
        )
        
    return keras_input_layers
