from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class Regressione():

    def __init__(self):
        self.linear = LinearRegression()
        self.lasso = Lasso(alpha=0.1) # parametro di penalizzazione
        self.ridge = Ridge(alpha=1.0) # più alto -> coefficenti più piccoli
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def split_data(self, data):
        X = data.drop(columns='mpg')

        y = data['mpg']

        # 80 train 20 test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size = 0.2,
            random_state = 42
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def regressione_lineare(self): # regressione normale minimizza media ^2
        self.linear.fit(self.X_train, self.y_train)

        y_pred = self.linear.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        return {
            "predizioni": y_pred,
            "coeff": self.linear.coef_,
            "intercetta": self.linear.intercept_,
            "MSE": mse,
            "R2": r2
        }
        # con i coefficenti vedo se effettivamente e' stato addestrato correttamente
        # siamo ancora livello molto uga buga

    def regressione_lasso(self): # aggiunge ulteriore penalizzazione per restringere coefficenti
        self.lasso.fit(self.X_train, self.y_train)

        y_pred = self.lasso.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        return {
            "predizioni": y_pred,
            "coeff": self.linear.coef_,
            "intercetta": self.linear.intercept_,
            "MSE": mse,
            "R2": r2
        }

    def regressione_ridge(self): # via di mezzo tra le 2 precedenti
        self.ridge.fit(self.X_train, self.y_train)

        y_pred = self.ridge.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        return {
            "predizioni": y_pred,
            "coeff": self.linear.coef_,
            "intercetta": self.linear.intercept_,
            "MSE": mse,
            "R2": r2
        }

