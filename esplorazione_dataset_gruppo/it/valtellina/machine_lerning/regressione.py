from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


class Regressione():

    def __init__(self):
        self.linear = LinearRegression()
        self.lasso = Lasso(self.set_alpha_L()) # parametro di penalizzazione
        self.ridge = Ridge(self.set_alpha_R()) # più alto -> coefficenti più piccoli
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def set_alpha_L(self, alpha_L=0.1):
        if alpha_L <= 0:
            print("Il parametro alpha per la Lasso deve essere maggiore di zero")
            alpha_L = 0.1
        return alpha_L
    
    def set_alpha_R(self, alpha_R=1):
        if alpha_R <= 0:
            print("Il parametro alpha per la Ridge deve essere maggiore di zero")
            alpha_R = 1.0
        return alpha_R

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
            "predizioni": y_pred.tolist(),
            "coeff": self.linear.coef_.tolist(),
            "intercetta": float(self.linear.intercept_),
            "MSE": float(mse),
            "R2": float(r2)
        }
        # con i coefficenti vedo se effettivamente e' stato addestrato correttamente
        # siamo ancora livello molto uga buga

    def regressione_lasso(self): # aggiunge ulteriore penalizzazione per restringere coefficenti
        
        scaler = StandardScaler()

        # fit SOLO sul train
        X_train_scaled = scaler.fit_transform(self.X_train)

        # transform sul test
        X_test_scaled = scaler.transform(self.X_test)

        self.lasso.fit(X_train_scaled, self.y_train)

        y_pred = self.lasso.predict(X_test_scaled)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        return {
            "predizioni": y_pred.tolist(),
            "coeff": self.linear.coef_.tolist(),
            "intercetta": float(self.linear.intercept_),
            "MSE": float(mse),
            "R2": float(r2)
        }

    def regressione_ridge(self): # via di mezzo tra le 2 precedenti

        scaler = StandardScaler()
        # fit SOLO sul train
        X_train_scaled = scaler.fit_transform(self.X_train)

        # transform sul test
        X_test_scaled = scaler.transform(self.X_test)


        self.ridge.fit(X_train_scaled, self.y_train)

        y_pred = self.ridge.predict(X_test_scaled)
        mse = mean_squared_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        return {
            "predizioni": y_pred.tolist(),
            "coeff": self.linear.coef_.tolist(),
            "intercetta": float(self.linear.intercept_),
            "MSE": float(mse),
            "R2": float(r2)
        }

