import glob
import os
import re
import sys
import argparse

folders = ['train', 'val', 'test', 'full']


def main(arguments):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path',
                        help='Expresión regular que hace match sobre los directorios de los modelos a probar.',
                        type=str)
    args = parser.parse_args(arguments)

    models = [x for x in glob.glob(args.path) if os.path.isdir(x)]
    with open("csv_out.csv", 'w') as csv_out:
        csv_out.write("MODELO;Accuracy;;;;Precission;;;;Recall;;;;F1 Score;;;\n"
                      ";Train;Val;Test;POST;Train;Val;Test;POST;Train;Val;Test;POST;Train;Val;Test;POST\n")
        for model in models:
            model_name = model.split('\\')[-1]
            accuracy, precission, recall, f1score = [], [], [], []
            for folder in folders:
                print(os.path.normpath(f'{model}\\{folder}*\\full_eeg\\*.txt'))
                file = glob.glob(os.path.normpath(f'{model}\\{folder}*\\full_eeg\\*.txt'))
                print(file)
                with open(file[0]) as class_report:
                    string = [i.split() for i in class_report.readlines()]
                    # print(string)
                    accuracy.append(string[5][1].replace('.', ','))
                    precission.append(string[7][2].replace('.', ','))
                    recall.append(string[7][3].replace('.', ','))
                    f1score.append(string[7][4].replace('.', ','))
            csv_out.write(f'{model_name}')
            for i in accuracy + precission + recall + f1score:
                csv_out.write(f';{i}')
            csv_out.write("\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
