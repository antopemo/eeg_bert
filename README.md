# Clasificación de pacientes de Parkinson a partir de su encefalograma aplicando Deep-Learning.
Este es el repositorio que aloja el código de mi Proyecto Fin de Grado.
Dicho proyecto consiste en la construcción de modelos neuronales basados en BERT para clasificar electroencefalogramas (EEG).

Contiene las funciones que se han usado para entrenar y probar los diferentes modelos 
que se han empleado. 

A continuación se hace una expliación del contenido del repo:

- **BertTraining.py**: Entrena un modelo BERT, con los parámetros especificados en la cabecera.
- **BertTraining_Zones.py**: Entrena un modelo BERT por Zonas, con los parámetros especificados en la cabecera.
- **predictions.py**: Es el archivo que se encarga de ejecutar todas las pruebas sobre los modelos que se han entrenado
- **aux_func**: Carpeta que contiene scripts auxiliares:

    - **confussion_matrix.py**: Permite generar las matrices de confusión.
    Script de http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    - **data_preprocess.py**: Contiene la clase Preprocessor que se encarga de generar los datasets para los modelos.
    - **data_reading.py**: Permite leer los archivos y refactorizarlos a arrays de NumPy para su posterior utilización. 
    También contiene las funciones de la ventana deslizante en forma de generadores.
    - **load_model.py**: Permite cargar modelos desde archivo.
    - **plot_model.py**: Permite pintar una representación del modelo.

- **BERT**: Carpeta que contiene el desarrollo de https://github.com/kpe/bert-for-tf2/tree/master/bert. 
Se ha modificado un archivo y se ha añadido uno nuevo:

    - **embeddings.py**: Se ha eliminado la parte en la que se convierte una palabra en vector y solamente se hace una transposición de los datos.
    
   - **splitter_layer.py**: Es la capa que distribuye los canales a cada BERT individual en el modelo por zonas.
   

    
     