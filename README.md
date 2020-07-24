# Clasificación de pacientes de Parkinson a partir de su encefalograma aplicando Deep-Learning.
Este es el repositorio que aloja el código de mi Proyecto Fin de Grado.
Dicho proyecto consiste en la construcción de modelos neuronales basados en BERT para clasificar electroencefalogramas (EEG).

Contiene las funciones que se han usado para entrenar y probar los diferentes modelos 
que se han empleado. 

## USO
El script ``eegbert.py`` es el punto de entrada a todo lo que se puede hacer con la librería.

Consta de 3 modos diferentes:
- ``-t`` o ``--training`` que se usa para entrenar un modelo en concreto

- ``-i`` o ``--inference`` que se usa para generar un archivo de clasificación de unos datos sobre un modelo

- ``-e`` o ``--evaluation`` que se usa para, a partir de un archivo de clasificación y (opcionalmente) uno que contenga
los valores reales, generar métricas de rendimiento y una matriz de confusión.

La lista completa de opciones se puede comprobar mediante el comando ``--help``.

A continuación detallamos un ejemplo de un entrenamiento, una inferencia con unos datos y una evaluación del rendimiento.

### Entrenamiento
Ponemos a entrenar un modelo de 40 canales especificado en el archivo de configuración, sin padding, con el conjunto de datos nuevos.

``eegbert.py -t --modelo 40_ch --nopad --paciente pre_post --prueba all``
    
### Inferencia
Usamos el modelo entrenado anteriormente para clasificar unos EEGs (especificados por regex).

``eegbert.py -i --path "RUTA_DEL_MODELO" --dataset "RUTA_DE_LOS_EEG\Sujeto*[!npy]" --out_file "inference_test.csv"``

### Evaluacion
A partir de las métricas obtenidas anteriormente se obtiene. Tags.csv es un archivo que en la primera columna contiene la ruta del eeg y en la segunda su tag (0 No Parkinson y 1 Parkinson).

``eegbert.py -e --in_file "RUTA\inference_test.csv" --real_file "RUTA\tags.csv" --mode full``

## DATOS
Los datos a los que hacemos referencia como los datos viejos se consiguen especificando en las opciones 
que sean pacientes ``pre``, ``FTI``, y su conjunto de test se evaluaba sobre ``test_pre_post``

Los datos nuevos usan pacientes ``pre_post``, como pruebas ``all`` y el conjunto test es ``test``