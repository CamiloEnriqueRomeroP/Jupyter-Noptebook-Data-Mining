from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import cm


# Lectura del data set
df = pd.read_csv('/Users/camilo/Library/CloudStorage/GoogleDrive-camiloeromerop@gmail.com/Mi unidad/Doctorado/Minería-de-datos/Introduction-to-python/churn.csv', sep=';')

# Preparación de los datos
df.columns = (
    df.columns
    .str.strip()                         # Elimina espacios alrededor
    .str.replace(' ', '_', regex=False) # Reemplaza espacios por _
    .str.replace(r'^_+', '', regex=True)# Elimina _ al inicio
    .str.replace('"', '', regex=False)
)
df.drop({"State","Area_Code"}, axis=1, inplace=True)
df = df.apply(lambda col: col.str.strip().str.replace('"', '', regex=False) if col.dtype == 'object' or pd.api.types.is_string_dtype(col) else col)
df['VoiceMail_Plan'] = df['VoiceMail_Plan'].map({'yes': 1, 'no': 0})
df['Inter_Plan'] = df['Inter_Plan'].map({'yes': 1, 'no': 0})
df['Churn'] = df['Churn'].map({'FALSE': 0, 'TRUE': 1})

# Separación del data set de entrenamiento y prueba con escalado
np.random.seed(42)
X = df[['Inter_Plan', 'VoiceMail_Plan',
       'No_of_Vmail_Mesgs', 'Total_Day_Min', 'Total_Day_calls',
       'Total_Evening_Min', 'Total_Evening_Calls',
       'Total_Int_Min', 'Total_Int_Calls',
       'No_of_Calls_Customer_Service']]
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenamiento del modelo
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train_scaled, y_train)

# Predicciones
y_pred = knn.predict(X_test_scaled)

# Reporte de métricas
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Probar múltiples modelos de clasificación
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train_scaled, X_test_scaled, y_train, y_test)
# Mostrar los resultados
print(models)

# Visualización con Seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - KNN')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()
