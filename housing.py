import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
'''

# Obtener la data de internet - Tal lo indicado en el libro
"""
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Invocar la funcion para descargar y descomprimir la data localmente
fetch_housing_data()

# Construyo Funcion para abrir el dataset ubicado en una carpeta diferente a donde esta el .py
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# Invoco la funcion para abrir el dataset
housing = load_housing_data()
"""

# Abro el dataset cuando está ubicado en la misma carpeta del .py
housing = pd.read_csv("housing.csv")



head_housing = housing.head()
print(head_housing)
housing.head() # Top five rows in the dataframe


housing.info() # Description of the data


# Categories related to column ocean_proximity - its maybe a Categorical attribute
housing["ocean_proximity"].value_counts()


# Summary of the numerical attributes
housing.describe()


# Plot Histogram to get another feel of the type of data
# solo para usar en Jupyter: %matplotlib inline
#import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()

# -----------------------------------------------
# Creating a Test Set
# Separando 20% de la data para Test - con random y split_train_test()
# La desventaja de utilizar este random es que cada vez que ejecute el modelo
# me arroja diferentes Set Test y por lo tanto el Train Set tambien es diferente
# esto conlleva a que de alguna manera se usa toda o buena parte de la data en
# el entrenamiento del modelo, lo cual no es bueno.

#import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    #print("test_set_size: ", test_set_size)        
    test_indices = shuffled_indices[:test_set_size]
    #print("test inidces: ", test_indices)    
    train_indices = shuffled_indices[test_set_size:]
    #print("train inidces: ", train_indices)    
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)

print(len(train_set), "train data +", len(test_set), "test data")

# Esta no es la forma correcta de determinar train y test sets
# ya que al actualizar el dataset se estaria mezclando data
# de train y de test, es decir, no se respetaria su clasificacion
# inicial

# --------------------------------------------

# Crear un Test Set utilizando Hash en un identificador unico de cada instancia del dataset
# Para el caso del dataset housing no se tiene una columna con identificador unico para cada instancia 
# Por lo que puede usarse el row index como el identificador - ID
# Otra opcion seria utilizar el feature mas estable como identificador unico (ej. las coordenadas)
# esto en caso de no poder garantizar que no se eliminen registros del dataset original o de 
# poder incluir nueva data al dataset al final de este 

'''
import hashlib

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing_with_id = housing.reset_index() # Adiciona una columna 'index'
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# Utilizando la nueva columna index, no puedo eliminar registros del dataset original, y
# las nuevas adiciones al dataset deben realizarse al final del dataset original.  
'''
# -----------------------------------

# Seleccion Random
# Como la data siempre va a ser la misma (no cambia ni se incrementa o disminuye) usamos la funcion
# train_test_split para usar seed o random state garantizando que siempre genere los mismos indices
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)
print(len(train_set), "Train Set data +", len(test_set), "Test Set data")

# ------------------------------

# Dado que el atributo "median income" es importante para predecir el "median housing prices"
# y para evitar sesgos en la informacion, para garantizar que el "test set" sea representativo
# de las varias categorias de ingresos en el dataset, creamos un atributo categorico para el income
# dado que este es un valor continuo
# divido el "median_income" por 1.5 para limitar las categorias y las categorias cuyo valor sea


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace = True)

"""
Where Parameters
----------
cond : boolean NDFrame, array-like, or callable
    Where `cond` is True, keep the original value. Where
    False, replace with corresponding value from `other`.
    If `cond` is callable, it is computed on the NDFrame and
    should return boolean NDFrame or array. The callable must
    not change input NDFrame (though pandas doesn't check it).

    .. versionadded:: 0.18.1
        A callable can be used as cond.
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

print(len(train_set), "Train Set data +", len(test_set), "Test Set data")
other : scalar, NDFrame, or callable
    Entries where `cond` is False are replaced with
    corresponding value from `other`.
    If other is callable, it is computed on the NDFrame and
    should return scalar or NDFrame. The callable must not
    change input NDFrame (though pandas doesn't check it).

    .. versionadded:: 0.18.1
        A callable can be used as other.

inplace : boolean, default False
    Whether to perform the operation in place on the data
axis : alignment axis if needed, default None
level : alignment level if needed, default None
errors : str, {'raise', 'ignore'}, default 'raise'
    - ``raise`` : allow exceptions to be raised
    - ``ignore`` : suppress exceptions. On error return original object

    Note that currently this parameter won't affect
    the results and will always coerce to a suitable dtype.

try_cast : boolean, default False
    try to cast the result back to the input type (if possible),
raise_on_error : boolean, default True
    Whether to raise on invalid data types (e.g. trying to where on
    strings)

    .. deprecated:: 0.21.0from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

print(len(train_set), "Train Set data +", len(test_set), "Test Set data")
    
    
When inplace=True is passed, the data is renamed in place (it returns nothing), so you'd use:
df.an_operation(inplace=True)

When inplace=False is passed (this is the default value, so isn't necessary), 
performs the operation and returns a copy of the object
"""
# --------------------------------------

housing["income_cat"].value_counts()

# --------------------------------------

# Seleccion con Stratified sampling de acuerdo con la nueva categoria "income_cat"

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
print ("Tamaño strat_train_set", len(strat_train_set))
print ("Tamaño strat_test_set", len(strat_test_set))
#print (strat_train_set)

#Proporcion de la categoria "income_cat" en el test_set
print("Proporcion con Sampling de la categoria income_cat en el test_set: ", (strat_test_set["income_cat"].value_counts() / len(strat_test_set) * 100))

# ------------------------------------------------------------

# De acuerdo con Seleccion Random

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

print(len(train_set), "Train Set data +", len(test_set), "Test Set data")


#Proporcion de la categoria "income_cat" en el test_set con seleccion de datos aleatoria
print("Proporcion RANDOM de la categoria income_cat en el test_set: ", (test_set["income_cat"].value_counts() / len(strat_test_set) * 100))

# -------------------------------------------------------------

#Proporcion de la categoria "income_cat" en el dataset completo
print ("Proporcion la categoria income_cat en el dataset completo: ", (housing["income_cat"].value_counts() / len(housing)*100))

# ------------------------------------------------------------

#housing.head() # Top five rows in the dataframe
strat_train_set.head()

# --------------------------------------------------------

# Removemos el atributo income_cat de strat_train_set y de strat_test_set para dejar el dataset
# en su estado original

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis = 1, inplace = True) # drop values from column income_cat
    
    # axis = 1: Recorre verticalmente por columnas (izquierda a derecha) ???
    # axis = 0: Recorre horizontalmente por filas (arriba hacia abajo) ???
# -----------------------------------------------------------

# housing.head() # Top five rows in the dataframe
# strat_train_set.head()

# -----------------------------------------------------------

# Creo copia del training set para no dañar el original.Lo llamo housing
housing = strat_train_set.copy()

#-------------------------------------------------------------
housing.head()

#-------------------------------------------------------------

# Dado que tenemos informacion geografica (latitud y longitud) creamos un scatterplot
# de la copia de strat_train_set y que llamé housing

#housing.plot(kind = "scatter", x = "longitude", y = "latitude")
"""housing.plot(kind = "scatter", x = "longitude", y = "latitude", title = "High density areas", grid = True, 
             alpha = "0.1") """

housing.plot(kind = "scatter", x = "longitude", y = "latitude", title = "High Density Areas", grid = True, 
             alpha = 0.4, s = housing["population"]/100, label = "population", figsize = (10,7), 
             c = "median_house_value", cmap = plt.get_cmap("jet"), colorbar = True)
plt.legend()

# El radio de cada circulo representa la poblacion de cada distrito (opcion s)
# El color representa el precio (opcion c)
# color map predefinido (opcion cmap) llamado jet, que va de azul a rojo

#-------------------------------------------------------------------------------------------

# Determino correlaciones
# Con el coeficiente de correlacion estandar (Pearsons r) entre cada par de atributos
# respecto al atributo median_house_value

corr_matrix = housing.corr()

corr_matrix["median_house_value"].sort_values(ascending = False)

#-------------------------------------------------------------------------------------------


