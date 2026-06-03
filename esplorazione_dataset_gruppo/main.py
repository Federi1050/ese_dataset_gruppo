# analisi esplorativi dei dataset e pulizia
# analisi varianza
# libreria scikit-lern
# praticamente la parte EDA
# normalizzare i dati
# metti anche il grafico
# fai in modo che sia impostabile con flask per mettere di valore e la sua risposta

from it.valtellina.datatset.dataset_manager import Ds_manager
from it.valtellina.flask.connection_manager import Flask_connection_manager

toggle = False
# true -> flask
# false -> console

if __name__ == "__main__":
    if toggle:
        app = Flask_connection_manager()
        app.run(debug=True)
    else:
        # crea manager e importa csv
        ds_mn = Ds_manager()

        ds_mn.clean_data()

        # stampa dataset
        print("mostra prime 5 righe")
        print(ds_mn.mostra_cont(5))

        # stampa shape e info del mio dataset
        print("mostra shape e info")
        print(ds_mn.struttura())

        # stampa describe del mio dataset
        print("mostra summery solo per numeri")
        print(ds_mn.summary())  # ovviamente solo per variabili numeriche

        # mostra quanti sono i missing values
        print("Numero di missing value per colonna:", ds_mn.missing_values())
        print("Missing value in %:", ds_mn.missing_percentage())
        print("Matrice di correlazione", ds_mn.correlation_matrix())

        # iniziamo a mostrare qualche grafico
        print(ds_mn.plot_correlation())
        print(ds_mn.plot_hist("weight")) # 1 colonna
        print(ds_mn.plot_distribution("horsepower")) # 1 colonna
        print(ds_mn.plot_scatter("cylinders", "displacement")) # 2 colonne
        # print(ds_mn.plot_box("")) # 1 colonna
        print(ds_mn.plot_counts("model_year")) # 1 colonna

        # Individuazione outlier:
        print("Numero outlier secondo IQR:", ds_mn.outliers_iqr_per_col())
        print("Numero outlier secondo Z score:", ds_mn.outliers_zscore_per_col())
        '''
        lo zeta score standarizza i dati, assumendo una distribuzione normale, e considera outliers
        tutti quei valori in modulo superiori a 3, perchè in teoria in una norm. stand. il 99,7%  rientra in questo range
        non funziona bene con i redditi, distr. non normale
        '''

        print(ds_mn.valori_stringhe())


