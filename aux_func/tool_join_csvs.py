import glob

with open(r"C:\Users\Ceiec01\OneDrive - UFV\PFG\Codigo\errores3.csv", 'a+') as final:
    final.write('Fichero;Modelo;NoParkinson;Parkinson\n')
    for file in glob.glob(r"C:\Users\Ceiec01\OneDrive - UFV\PFG\Codigo\nuevos_modelos*"):
        with open(file) as the_file:
            the_file.readline()
            for line in the_file:
                final.write(line.replace('.', ','))