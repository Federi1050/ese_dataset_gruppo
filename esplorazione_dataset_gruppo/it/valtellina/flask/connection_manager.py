from flask import Flask

class Flask_connection_manager():
    def __init__(self):
        self.app = Flask(__name__)
        self.__register_routes()

    def __register_routes(self):
        @self.app.route("/")
        def home():
            return "Ciao da flask"


    def run(self, **kwargs):
        self.app.run(**kwargs)

        #**kwargs significa accetta un numero variabile di argomenti