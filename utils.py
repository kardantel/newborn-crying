from sklearn.metrics import confusion_matrix, classification_report
import scipy.fftpack as fourier
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class Utils:
    def trim(self, dataframe, startIndex=0, endIndex=0):
        '''
        Allows to separate the dataframe based on an initial and a final index.
    
        ## Parameters
        dataframe: array-like of shape (n_samples, n_features)
            Original dataset to work with. Can be for example a list, or an array.
        startIndex: int, default=`0`
        endIndex: int, default=`0`
        '''
        dataframe = dataframe[[i for i in range(startIndex, endIndex)]]
        dataframe.columns = np.arange(dataframe.shape[1])
        return dataframe

    def fftSignal(self, inputsignal, rate):
        '''
        Calcula la transformada de Fourier.

        ## Parámetros
        inputsignal: la señal para la que se calculará la transformada.
        rate: frecuencia de muestreo.
        '''
        # Calcular la transformada de Fourier (lista de números complejos)
        signalf = fourier.fft(inputsignal)
        # La mitad de la lista fft (simetría de señal real)
        h = len(signalf) // 2
        signalf2 = abs(signalf[:(h - 1)])
        # Vector de frecuencias
        f = np.linspace(start=0, stop=rate / 2, num=len(signalf2))

        return f, signalf2

    def getBw(self, signalf, f, u):
        '''
        Calcula el ancho de banda de la señal.

        ## Parámetros
        signalf, f: módulo de la transformada de Fourier de la señal.
        u: umbral para determinar el Bw.
        '''
        # Potencia total de la señal
        total = np.sum(signalf)
        cont = 0
        pot = 0
        while (pot < u * total):
            pot = np.sum(signalf[0:cont])
            cont += 1
        bw = 0.5 * f[cont - 1] + 0.5 * f[cont]

        return bw

    def xData(self, df, sr):
        '''
        Calcula la media, la mediana, la desviación estándar y el ancho de banda,
        de cada una de las muestras del dataset de entrada, devolviendo dichos
        valores en un dataset de tipo Pandas.

        ## Parámetros
        df: dataset al que se le calcularán características estadísticas.
        '''
        media = list()
        mediana = list()
        std = list()
        bw = list()
        for i in range(0, len(df)):
            print("\r     Registro: ", i + 1, "/", len(df), end="")
            idex_val = df.loc[df.index[i]].values
            # Cálculos estadísticos
            mean_val = np.mean(idex_val)  # Cálculo de la media.
            media.append(mean_val)
            median_val = np.median(idex_val)  # Cálculo de la mediana.
            mediana.append(median_val)
            std_val = np.std(idex_val)  # Cálculo de la desviación estándar.
            std.append(std_val)
            # Cálculos en el dominio de la frecuencia
            # Cálculo del transformada de Fourier.
            fft_val = self.fftSignal(idex_val, sr)
            # Cálculo del ancho de banda.
            bw_val = self.getBw(fft_val[1], fft_val[0], 0.9)
            bw.append(bw_val)

        return pd.DataFrame([media, mediana, std, bw]).T

    def setTag(self, dataset):
        '''
        Devuelve el dataset con las etiquetas de las emociones.

        ## Parámetros
        dataset: dataset al que se le agregará la columna con los sentimientos.
        '''
        list_nom = []
        for n in dataset.index:
            list_nom.append(n[0:5])
        nombres = pd.DataFrame(
            list_nom, columns=['tag'], index=dataset.index, dtype="category")

        return dataset.assign(tag=nombres['tag'])

    def getMetrics(self, y_test, y_pred):
        '''
        Imprime el reporte con métricas de validación para clasificación y la matriz de confusión.

        y_test: valor de prueba real de la etiqueta de salida.
        y_pred: valor predicho al ejecutar el método 'predict()'.
        '''
        print(classification_report(y_test, y_pred))

        conMatrix = confusion_matrix(y_test, y_pred)
        # Definimos los valores para cada grupo.
        class_names = np.unique(y_test)
        fig, ax = plt.subplots()
        # Creamos un vector con la cantidad de valores que hay en 'class_names' desde 0.
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)

        sns.heatmap(pd.DataFrame(conMatrix), annot=True, cmap="Blues_r", fmt="g")
        plt.tight_layout()
        plt.ylabel("Etiqueta Actual")
        plt.xlabel("Etiqueta de Predicción")

    def prefix(self, t):
        '''
        Forma práctica de agregar prefijos generalizados tanto a índices como a columnas.
        '''
        def p(x):
            return f"{t}{int(x)+1}"
        return p
