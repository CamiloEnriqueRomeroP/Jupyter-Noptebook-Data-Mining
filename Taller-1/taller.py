from scipy.io import arff
import pandas as pd

# Se realiza la lectura del archivo arff y se crea el dataFrame df para almacenarlo
data, meta = arff.loadarff("churn.arff")
df = pd.DataFrame(data)
# Si el dataset contiene celdas en formato byte se codifican en formato utf-8
df = df.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)

for column1 in df:
    for column2 in df:
        for i in range(len(column1)):
            if df[column1].iloc[i] != df[column2].iloc[i]:
                nuevo_nombre = f"{column1}_by_{column2}"
                df[nuevo_nombre] = df[column1].iloc[i]/df[column2].iloc[i]
                print("Valores iguales")
            else:
                nuevo_nombre = f"{column1}_by_{column2}"
                df[nuevo_nombre] = 1
print(df.head())
