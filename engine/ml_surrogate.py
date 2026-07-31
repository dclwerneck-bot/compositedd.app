import numpy as np
try:
    from engine.clt import Lamina, Laminate
    from engine.optimization import DDOptimizer
except ImportError:
    from clt import Lamina, Laminate
    from optimization import DDOptimizer

try:
    from sklearn.ensemble import RandomForestRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class MLSurrogate:
    """Modelo de Aprendizado de Máquina (Surrogate Model) treinado offline para recomendação instantânea de DD."""
    def __init__(self, lamina: Lamina):
        self.lamina = lamina
        self.optimizer = DDOptimizer(lamina)
        self.is_trained = False
        self.model_phi = None
        self.model_psi = None

    def train_surrogate(self, n_samples=300):
        """Gera um banco de dados sintético por CLT e treina 2 modelos RandomForest para prever Phi e Psi instantaneamente."""
        if not HAS_SKLEARN:
            return False

        # Gera combinações aleatórias de razões de carga Nx, Ny, Nxy e razão de rigidez desejada
        X_train = []
        y_phi = []
        y_psi = []

        np.random.seed(42)
        ratios = np.linspace(0.1, 5.0, 50)

        for r in ratios:
            res = self.optimizer.optimize_for_stiffness_ratio(target_ratio=r, step=5.0)
            X_train.append([r])
            y_phi.append(res["phi"])
            y_psi.append(res["psi"])

        X_train = np.array(X_train)
        y_phi = np.array(y_phi)
        y_psi = np.array(y_psi)

        self.model_phi = RandomForestRegressor(n_estimators=30, random_state=42)
        self.model_psi = RandomForestRegressor(n_estimators=30, random_state=42)

        self.model_phi.fit(X_train, y_phi)
        self.model_psi.fit(X_train, y_psi)
        self.is_trained = True
        return True

    def predict_dd(self, target_stiffness_ratio):
        """Previsão em milissegundos dos ângulos Phi e Psi ideais para a razão de rigidez solicitada."""
        if not self.is_trained or not HAS_SKLEARN:
            # Fallback direto via otimizador
            res = self.optimizer.optimize_for_stiffness_ratio(target_ratio=target_stiffness_ratio, step=5.0)
            return res["phi"], res["psi"], False
        
        pred_phi = float(self.model_phi.predict([[target_stiffness_ratio]])[0])
        pred_psi = float(self.model_psi.predict([[target_stiffness_ratio]])[0])
        
        # Arredonda para o passo de 0.5 deg
        pred_phi = round(pred_phi * 2) / 2.0
        pred_psi = round(pred_psi * 2) / 2.0
        
        return pred_phi, pred_psi, True
