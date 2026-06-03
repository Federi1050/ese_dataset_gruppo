from flask import Flask, jsonify
from it.valtellina.datatset.dataset_manager import Ds_manager
from it.valtellina.machine_lerning.regressione import Regressione

class Flask_connection_manager():
    def __init__(self):
        self.app = Flask(__name__)
        self.__register_routes()

        self.ds_mn = Ds_manager()
        self.reg = Regressione()

        lista = ["acceleration", "origin"]
        self.ds_mn.clean_data(lista)

    def __register_routes(self):
        @self.app.route("/")
        def home():
            return "Ciao da flask"

        @self.app.route("/info_csv")
        def info_csv():
            obj = {
                "head": self.ds_mn.mostra_cont(5),
                "shape and info": self.ds_mn.struttura(),
                "summary num": self.ds_mn.summary(),
                "missing values": self.ds_mn.missing_values(),
                "missing v in %": self.ds_mn.missing_percentage(),
                "mat correlazione": self.ds_mn.correlation_matrix()
            }
            return jsonify(obj)

        @self.app.route("/plots")
        def plots():
            obj 





























    def run(self, **kwargs):
        self.app.run(**kwargs)

        #**kwargs significa accetta un numero variabile di argomenti