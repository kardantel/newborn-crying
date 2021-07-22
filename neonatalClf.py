import numpy as np
import pandas as pd
import math
import librosa
import matplotlib.pyplot as plt
from utils import Utils
import os


def neonatalClf(audio_list, file_name, toCsv=False, parts=0, sr=None):
    '''
    Function that receives a list of audios in WAV format and is in pandas
    dataframe format where the columns are the amount of data for each audio
    and the rows are each of the individual audio samples. To load the audio
    files, the Librosa library is used, which delivers the values sampled and
    normalized between 0 and 1.
    Returns a dataframe with the separated audios, special statistics and the
    corresponding prediction label.
    
    ## Parameters
    audio_list: array-like of shape (n_samples, n_features)
        List with the names of the audios to be saved in the dataframe.
        Can be for example a list, or an array.
    file_name: str
        Name of the created CSV file.
    toCsv: Boolean [optional], defaults to `None`.
        If True, it saves in csv format in the directory chosen in
        the folder named 'csv' that must be created before execution.
        The default value 'toCsv = False', prevents conversion`
    parts: int, default=`0`
        Amount of cuts to be made, in order to increase rows and reduce columns.
    sr: int, default=`None`.
        Sampling frequency of audio files.
    '''
    # Variable initialization.
    file_size = []
    df = pd.DataFrame()
    utils = Utils()

    # The list of audios is scrolled.
    print("Creating the dataset...")
    for i in audio_list:
        print("\r     File:", audio_list.index(i) + 1, "/", len(audio_list), end="",)
        # The normalized wav files are loaded.
        sonido, sr = librosa.load(f'wav/{i}', sr=sr)
        # The indices where the audio starts and ends are calculated and the corresponding subarray is obtained.
        sonido = sonido[np.nonzero(sonido)[0][0]:np.nonzero(sonido)[0][-1]]

        # The file size is added to the file_size list.
        file_size.append(len(sonido))

        # Each of the audios are added to the dataframe.
        df = pd.concat([df, pd.Series(sonido, name=i[0:-4])], axis=1)

    print("\nHomogenizing dataframe dimensions...")
    # The dataframe is transposed to have the information of each file as rows and the missing data is filled.
    df = df.fillna(0).T
    for index, row in df.iterrows():
        print("\r     Audio record:", df.index.get_loc(index) + 1, "/", len(df), end="")
        # Audio is trimmed to its maximum size
        sonido = row[0:file_size[df.index.get_loc(index)]]
        # The sound is replicated until the maximum file size is exceeded.
        sonido = np.tile(sonido, max(file_size) // file_size[df.index.get_loc(index)] + 1)
        # New audio is trimmed to the same size as the largest original audio.
        df.loc[index] = sonido[0:max(file_size)]

    print("\nSeparating audios...")
    sizes = df.shape[1] // parts
    start = 0
    end = sizes
    df_splitted = []
    for i in range(parts):
        print("\r     Part:", i + 1, "/", parts, end="")
        new_df = utils.trim(df, start, end)
        start = end + 1
        end = end + sizes
        if(i):
            new_df.index = [ind + '_' + (i.__str__()) for ind in df.index]
        df_splitted.append(new_df)
    new_df_splitted = pd.concat(df_splitted, axis=0)
    new_df_splitted = new_df_splitted.iloc[:, :-1]

    print("\nCalculating statistics...")
    # Se calculan, en un nuevo dataframe los valores estadísticos de la media, mediana y desviación estándar de cada fila.
    sig_dt = utils.xData(new_df_splitted, sr)
    # Se cambian los nombres de las columnas y las filas del dataframe con valores estadísticos.
    sig_dt = sig_dt.rename(columns={0: 'mean', 1: 'median', 2: 'std', 3: 'bw'}).set_axis(new_df_splitted.index, axis='index')
    # Se concatenan los dataframes y se renombran los índices
    new_df_splitted = pd.concat([new_df_splitted, sig_dt], axis=1)
    new_df_splitted = utils.setTag(new_df_splitted)

    if toCsv:
        print("\nSaving CSV file")
        df.to_csv(f'csv/{file_name}', index=False, chunksize=100)

    return new_df_splitted


if __name__ == "__main__":
    # find . -name "*.DS_Store" -type f -delete
    listaAudios = os.listdir("wav/")
    neonatalClf(listaAudios, "neo_audios.csv", toCsv=False, parts=25, sr=12000)
