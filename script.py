# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Distribuzione Statistica delle Misure del Periodo di un Pendolo
# Analisi dei dati con i pacchetti relativi a SciPy
# 
# ## Setup dell'ambiente di lavoro

# %%
get_ipython().system(' pip install tikzplotlib')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as numpy
from scipy.optimize import curve_fit
import math
import tikzplotlib

# %% [markdown]
# ## Importazione dei dati
# Importiamo i dati da un file CSV ottenuto a partire da quello fornito e dividiamo le tre serie di misure in tre variabili distinte, in quanto hanno lunghezze differenti e non sono correlate per riga.

# %%
pendolo_df = pd.read_csv('https://raw.githubusercontent.com/persello/pendolo/master/Dati_Pronti.csv', sep=';', decimal=',')

pendolo30 = pendolo_df['Periodo30'].dropna()
pendolo100 = pendolo_df['Periodo100'].dropna()
pendolo300 = pendolo_df['Periodo300'].dropna()

# %% [markdown]
# ## Calcolo di media e deviazione standard

# %%
print("Le medie sono:", pendolo30.mean(), "per N =", pendolo30.size, ",", pendolo100.mean(), "per N =", pendolo100.size,"e", pendolo300.mean(), "per N =", pendolo300.size)
print("Le deviazioni sono:", pendolo30.std(), "per N =", pendolo30.size, ",", pendolo100.std(), "per N =", pendolo100.size,"e", pendolo300.std(), "per N =", pendolo300.size)

# %% [markdown]
# ## Filtraggio delle misure
# Vengono rimosse le misure fuori da $[\bar{x}-3\sigma_c, \bar{x}+e\sigma_c]$.

# %%
def filtro_3sigma(data):
    return data[data.between(data.mean() - 3*data.std(), data.mean() + 3*data.std())].dropna()

pendolo30f = filtro_3sigma(pendolo30)
pendolo100f = filtro_3sigma(pendolo100)
pendolo300f = filtro_3sigma(pendolo300)

print("Sono state rimosse", pendolo30.size - pendolo30f.size, "righe, nuovo N =", pendolo30f.size)
print("Sono state rimosse", pendolo100.size - pendolo100f.size, "righe, nuovo N =", pendolo100f.size)
print("Sono state rimosse", pendolo300.size - pendolo300f.size, "righe, nuovo N =", pendolo300f.size)

# %% [markdown]
# ## Calcolo delle nuove medie e deviazioni
# Ricalcoliamo medie e deviazioni dopo il filtraggio, prima di calcolare la deviazione standard della media e di calcolare il fit.

# %%
print("Le nuove medie sono:", pendolo30f.mean(), "per N =", pendolo30f.size, ",", pendolo100f.mean(), "per N =", pendolo100f.size,"e", pendolo300f.mean(), "per N =", pendolo300f.size)
print("Le nuove deviazioni sono:", pendolo30f.std(), "per N =", pendolo30f.size, ",", pendolo100f.std(), "per N =", pendolo100f.size,"e", pendolo300f.std(), "per N =", pendolo300f.size)

# %% [markdown]
# ## Calcolo della deviazione standard della media

# %%
print("Le deviazioni standard delle medie sono:", pendolo30f.std()/math.sqrt(pendolo30f.size), "per N =", pendolo30f.size, ",", pendolo100f.mean()/math.sqrt(pendolo100f.size), "per N =", pendolo100f.size,"e", pendolo300f.mean()/math.sqrt(pendolo300f.size), "per N =", pendolo300f.size)

# %% [markdown]
# ## Calcolo del fit ottimale della funzione di distribuzione di Gauss

# %%
plt.style.use("ggplot")

def calcolaFit(data):

    # Creo un istogramma dato dai confini dei bin e dalle relative frequenze
    bin_count = (data.max() - data.min()) / (data.std()/2.2)
    histogram, bin_edges = numpy.histogram(data, bins=int(round(bin_count)))

    # Dimensione di un bin
    print("Dimensione del bin è", bin_edges[1]-bin_edges[0])

    # Calcolo i centri dei bin
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

    # Definisco la funzione Gaussiana con A parametro
    def gauss(x, A):
        return A*numpy.exp(-(x-data.mean())**2/(2*data.std()**2))

    fitted_params, var_matrix = curve_fit(gauss, bin_centers, histogram)

    # Stampo i risultati
    print("Parametro trovato: A =", fitted_params[0])

    # Faccio dei grafici
    gauss_x = numpy.linspace(1.2, 1.45, 1000)
    gauss_y = gauss(gauss_x, fitted_params[0])

    plt.plot(gauss_x, gauss_y)
    plt.hist(data, bins=bin_edges) 
    plt.xlabel("Periodo (s)")
    plt.ylabel("Frequenza della misura")
    plt.title("$N=" + str(data.size) + ", \\bar{x}=" + str(round(data.mean(), 4)) + ", \\sigma_c=" + str(round(data.std(), 4)) + "$")

    # Esporta file TikZ
    tikzplotlib.save("chart" + str(data.size) + ".tikz")

    plt.show()
    plt.close()


calcolaFit(pendolo30f)
calcolaFit(pendolo100f)
calcolaFit(pendolo300f)


# %%


