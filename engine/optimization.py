import numpy as np
try:
    from engine.clt import Lamina, Laminate
    from engine.failure import FailureAnalysis
except ImportError:
    from clt import Lamina, Laminate
    from failure import FailureAnalysis

class DDOptimizer:
    """Otimizador para laminados Double-Double [+-Phi / +-Psi]_n."""
    def __init__(self, lamina: Lamina, n_repeats=2):
        self.lamina = lamina
        self.n_repeats = n_repeats

    def build_dd_angles(self, phi, psi):
        """Constrói a sequência simétrica de ângulos DD."""
        sub = [phi, -phi, psi, -psi]
        full_sub = sub * self.n_repeats
        # Simétrico
        return full_sub + full_sub[::-1]

    def optimize_for_stiffness_ratio(self, target_ratio=1.0, step=2.5):
        """Busca o par (Phi, Psi) que melhor atinge a razão de rigidez Ex/Ey desejada."""
        best_phi, best_psi = 0.0, 0.0
        best_error = 1e9
        best_Ex, best_Ey = 0.0, 0.0
        
        angles_range = np.arange(0.0, 90.1, step)
        
        for phi in angles_range:
            for psi in angles_range:
                angles = self.build_dd_angles(phi, psi)
                lam = Laminate(self.lamina, angles)
                ratio = lam.Ex / lam.Ey
                error = abs(ratio - target_ratio)
                
                if error < best_error:
                    best_error = error
                    best_phi = phi
                    best_psi = psi
                    best_Ex = lam.Ex
                    best_Ey = lam.Ey
                    
        return {
            "phi": best_phi,
            "psi": best_psi,
            "Ex": best_Ex / 1e9,
            "Ey": best_Ey / 1e9,
            "ratio": best_Ex / best_Ey,
            "target_ratio": target_ratio
        }

    def optimize_for_max_load(self, Nx, Ny, Nxy, step=2.5):
        """Busca o par (Phi, Psi) que maximiza a Margem de Segurança de Tsai-Wu para as cargas aplicadas."""
        best_phi, best_psi = 0.0, 0.0
        max_margin = -1e9
        best_FI = 1e9
        
        angles_range = np.arange(0.0, 90.1, step)
        
        for phi in angles_range:
            for psi in angles_range:
                angles = self.build_dd_angles(phi, psi)
                lam = Laminate(self.lamina, angles)
                fa = FailureAnalysis(lam)
                res = fa.evaluate_laminate_failure(Nx, Ny, Nxy)
                
                if res["min_margin"] > max_margin:
                    max_margin = res["min_margin"]
                    best_FI = res["max_FI"]
                    best_phi = phi
                    best_psi = psi
                    
        return {
            "phi": best_phi,
            "psi": best_psi,
            "max_margin": max_margin,
            "min_FI": best_FI,
            "is_safe": best_FI < 1.0
        }
