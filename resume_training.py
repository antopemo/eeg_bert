import glob

from BertTraining import *

if __name__ == "__main__":
    model_path = "C:\\Users\\Ceiec01\\OneDrive - UFV\\PFG\\Codigo\\checkpoints\\BERT-20200225-121605"

    prepro = Preprocessor(batch_size,
                          window_width,
                          window_steps,
                          prueba=0,
                          limpio=0,
                          paciente=1,
                          channels=channels,
                          transpose=True,
                          output_shape=out_shape
                          )
    weights = glob.glob(model_path + "/training_weights/weights.*.hdf5")
    model = tf.keras.models.load_model(weights[-1], custom_objects={'BertModelLayer': BertModelLayer})

    dataset, train_dataset, test_dataset, val_dataset = Preprocessor(batch_size,
                                                                     window_width,
                                                                     window_steps,
                                                                     prueba=0,
                                                                     limpio=0,
                                                                     paciente=1,
                                                                     channels=channels,
                                                                     transpose=True,
                                                                     output_shape=out_shape
                                                                     ).classification_tensorflow_dataset()

    checkpoint_path = os.path.join(model_path, "training_weights\\weights_2.{epoch:02d}-{val_loss:.2f}.hdf5")
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_freq='epoch',
                                                     verbose=1)

    log_dir = ".log\\eegs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                       histogram_freq=1,
                                                       write_graph=True,
                                                       write_images=True,
                                                       update_freq='epoch',
                                                       profile_batch=2,
                                                       )
    total_epoch_count = 10
    model.fit(x=train_dataset,
              validation_data=val_dataset,
              shuffle=True,
              epochs=total_epoch_count,
              callbacks=[keras.callbacks.EarlyStopping(monitor="BinaryCross", patience=20, restore_best_weights=True),
                         tensorboard_callback, cp_callback])

    model.evaluate(x=test_dataset,
                   callbacks=[tensorboard_callback])
