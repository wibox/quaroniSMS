import tensorflow as tf
import tensorflow_model_optimization as tfmot
import os
import zipfile
from preprocessing import LABELS

def _log_header_to_csv(filename : str, header : str) -> bool:
    completed = False
    try:
        with open(f"results/{filename}", "w") as header_fp:
            header_fp.write(header + "\n")
        completed = True
    except Exception as e:
        print(e.format_exc())
    finally:
        return completed

def _log_output_to_csv(filename : str, content : str) -> bool:
    completed = False
    try:
        with open(f"results/{filename}", "a") as log_fp:
            log_fp.write(content + "\n")
        completed = True
    except Exception as e:
        print(e.format_exc())
    finally:
        return completed
    
def get_rnn():

    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=(32,32)))

    # model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False)))
    model.add(tf.keras.layers.SimpleRNN(128, return_sequences=True, activation="relu"))
    model.add(tf.keras.layers.SimpleRNN(128, return_sequences=False, activation="relu"))

    model.add(tf.keras.layers.Dense(128, activation=None))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Dense(64, activation=None))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Dense(3, activation=None))
    model.add(tf.keras.layers.Softmax())

    return model

def get_cnn(SHAPE, alpha, num_hidden_layers):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=SHAPE))
    for layer_counter in range(num_hidden_layers):
        if layer_counter == 0:
            model.add(tf.keras.layers.Conv2D(filters=128*alpha, kernel_size=[3, 3], strides=[2, 2],
                use_bias=False, padding='valid'))
        else:
            model.add(tf.keras.layers.Conv2D(filters=128*alpha, kernel_size=[3, 3], strides=[1, 1],
                use_bias=False, padding='same'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.ReLU())
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(units=len(LABELS)))
    model.add(tf.keras.layers.Softmax())
    
    return model

def compile_pruning_model(model, epoch, dim, i_lr, e_lr):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    begin_step = int(dim * epoch * 0.2)
    end_step = int(dim * epoch)
    final_sparsity=0.70
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.20,
            final_sparsity=final_sparsity,
            begin_step=begin_step,
            end_step=end_step
        )
    }

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    loss = tf.losses.SparseCategoricalCrossentropy(from_logits=False)

    initial_learning_rate = i_lr
    end_learning_rate = e_lr

    linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        end_learning_rate=end_learning_rate,
        decay_steps=dim * epoch,
    )
    optimizer = tf.optimizers.Adam(learning_rate=linear_decay)
    metrics = [tf.metrics.SparseCategoricalAccuracy()]
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    model_for_pruning.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model_for_pruning, callbacks

def get_model_statistics(history):
    training_loss = history.history['loss'][-1]
    training_accuracy = history.history['sparse_categorical_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_sparse_categorical_accuracy'][-1]

    return training_loss, training_accuracy, val_loss, val_accuracy

def convert_zip_save_model(model_for_pruning, idx):

    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

    print("Saving model to ./saved_models/")
    saved_model_dir = f'./saved_models/{idx}'
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    model_for_export.save(saved_model_dir)

    # model conversion to tf-lite format
    print("Converting model to TF-Lite format...")
    MODEL_NAME = idx
    ZIPPED_MODEL_NAME = MODEL_NAME
    converter = tf.lite.TFLiteConverter.from_saved_model(f'./saved_models/{MODEL_NAME}')
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    # saving tf-lite formatted model
    print("Saving TF-Lite model to ./tflite_models/")
    tflite_models_dir = './tflite_models'
    if not os.path.exists(tflite_models_dir):
        os.makedirs(tflite_models_dir)
    tflite_model_name = os.path.join(tflite_models_dir, f'{MODEL_NAME}.tflite')
    with open(tflite_model_name, 'wb') as fp:
        fp.write(tflite_model)

    # save the zipped model
    print("Zipping and saving the model to ./zipped_models/")
    if not os.path.exists("./zipped_models"):
        os.makedirs("./zipped_models")
    with zipfile.ZipFile(f'{os.path.join("./zipped_models",str(ZIPPED_MODEL_NAME))}.zip', 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(tflite_model_name)

    return MODEL_NAME, ZIPPED_MODEL_NAME