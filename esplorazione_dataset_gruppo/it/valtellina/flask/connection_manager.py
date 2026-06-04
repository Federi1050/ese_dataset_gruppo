from flask import Flask, jsonify, render_template_string, request
from it.valtellina.datatset.dataset_manager import Ds_manager
from it.valtellina.machine_lerning.regressione import Regressione
from io import BytesIO
import base64
import matplotlib.pyplot as plt

class Flask_connection_manager():
    def __init__(self):
        self.app = Flask(__name__)
        self.__register_routes()

        self.ds_mn = Ds_manager()
        self.reg = Regressione()

        lista = ["acceleration", "origin"]
        self.ds_mn.clean_data(lista)

        self.reg.split_data(self.ds_mn.data)

    def __register_routes(self):
        @self.app.route("/")
        def home():
            return "Ciao da flask"

        @self.app.route("/info_csv")
        def info_csv():
            dt = {
                "head": self.ds_mn.mostra_cont(5).to_json(),
                "shape and info": self.ds_mn.struttura(),
                "summary num": self.ds_mn.summary().to_json(),
                "missing values": self.ds_mn.missing_values().to_json(),
                "missing v in %": self.ds_mn.missing_percentage().to_json(),
                "mat correlazione": self.ds_mn.correlation_matrix().to_json()
            }
            return jsonify(dt)

        @self.app.route("/plots")
        def plots():
            figs = [
                self.ds_mn.plot_correlation(),
                self.ds_mn.plot_hist("weight"),  # 1 colonna
                self.ds_mn.plot_distribution("horsepower"),  # 1 colonna
                self.ds_mn.plot_scatter("cylinders", "displacement"),  # 2 colonne
                # ds_mn.plot_box(""), # 1 colonna
                self.ds_mn.plot_counts("model_year")  # 1 colonna
            ]

            images = []

            for fig in figs:
                img = BytesIO()
                fig.savefig(img, format="png", bbox_inches="tight")
                img.seek(0)

                encoded = base64.b64encode(img.getvalue()).decode()
                images.append(encoded)

                plt.close(fig)

            html = """
               <h1>Plots</h1>
               {% for img in images %}
                   <img src="data:image/png;base64,{{img}}" style="margin:10px;">
               {% endfor %}
               """

            return render_template_string(html, images=images)

        @self.app.route("/outlier")
        def outlier():
            dt = {
                "outl_iqr": self.ds_mn.outliers_iqr_per_col(),
                "outl_zscore": self.ds_mn.outliers_zscore_per_col()
            }
            return jsonify(dt)

        @self.app.route("/regressione", methods=["POST"])
        def regressione():
            data = request.get_json()

            metodo_regressione = data.get('metodo_regressione')
            alpha = data.get('alpha')

            try:
                metodo_regressione = int(metodo_regressione)
            except ValueError:
                return jsonify({"error": "non numerico"})

            # Controlli base
            if metodo_regressione not in [1, 2, 3]:
                return jsonify({"error": "metodo_regressione deve essere 1, 2 o 3"})

            if alpha is None:
                return jsonify({"error": "alpha è obbligatorio"})

            try:
                alpha = float(alpha)
            except ValueError:
                return jsonify({"error": "alpha deve essere numerico"})

            if metodo_regressione == 1:
                ris = self.reg.regressione_lineare()
            elif metodo_regressione == 2:
                self.reg.set_alpha_L(alpha)
                ris = self.reg.regressione_lasso()
            else:
                self.reg.set_alpha_R(alpha)
                ris = self.reg.regressione_ridge()

            return jsonify(ris)

        @self.app.route("/predizione", methods=["POST"])
        def predizione():
            # incompleta
            data = request.get_json()
            numero = data.get('numero')
            try:
                numero = int(numero)
            except ValueError:
                return jsonify({"error": "non hai inserito un numero"})

            return jsonify({"numero": numero})



























    def run(self, **kwargs):
        self.app.run(**kwargs)

        #**kwargs significa accetta un numero variabile di argomenti