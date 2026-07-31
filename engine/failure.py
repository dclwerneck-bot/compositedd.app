import numpy as np
try:
    from engine.clt import Laminate, Lamina
except ImportError:
    from clt import Laminate, Lamina

class FailureAnalysis:
    """Implementa critérios de falha de laminados compósitos (Tsai-Wu, Tsai-Hill, Máxima Tensão)."""
    def __init__(self, laminate: Laminate):
        self.laminate = laminate
        self.lamina = laminate.lamina
        
    def evaluate_ply_stresses(self, Nx, Ny, Nxy):
        """Dada a carga no plano (N/m), calcula deformaçoes e tensoes em cada camada nos eixos globais e locais."""
        N = np.array([Nx, Ny, Nxy], dtype=float)
        # Deformação no plano médio eps0 = a * N (assumindo B = 0 ou desprezando flexão pura)
        eps0 = self.laminate.a @ N
        
        ply_stresses_local = []
        
        for k in range(self.laminate.N):
            theta = self.laminate.angles[k]
            rad = np.radians(theta)
            m = np.cos(rad)
            n = np.sin(rad)
            
            # Matriz de transformação de deformação T_eps
            # [eps1, eps2, gamma12/2]^T = T * [epsx, epsy, gammaxy/2]^T
            Qbar = self.laminate.Qbars[k]
            sigma_global = Qbar @ eps0
            
            # Transformação de tensões do global (xy) para o local (12)
            # sigma1 = s_x*m^2 + s_y*n^2 + 2*t_xy*m*n
            # sigma2 = s_x*n^2 + s_y*m^2 - 2*t_xy*m*n
            # tau12  = (s_y - s_x)*m*n + t_xy*(m^2 - n^2)
            sx, sy, txy = sigma_global[0], sigma_global[1], sigma_global[2]
            
            s1 = sx * m**2 + sy * n**2 + 2 * txy * m * n
            s2 = sx * n**2 + sy * m**2 - 2 * txy * m * n
            t12 = (sy - sx) * m * n + txy * (m**2 - n**2)
            
            ply_stresses_local.append((s1, s2, t12))
            
        return ply_stresses_local

    def tsai_wu(self, s1, s2, t12):
        """Retorna o índice de falha de Tsai-Wu. FI < 1 => Seguro; FI >= 1 => Falha."""
        XT, XC = self.lamina.XT, self.lamina.XC
        YT, YC = self.lamina.YT, self.lamina.YC
        S = self.lamina.S
        
        F1 = 1.0/XT - 1.0/XC
        F11 = 1.0 / (XT * XC)
        F2 = 1.0/YT - 1.0/YC
        F22 = 1.0 / (YT * YC)
        F66 = 1.0 / (S**2)
        F12 = -0.5 * np.sqrt(F11 * F22)
        
        FI = F1*s1 + F2*s2 + F11*s1**2 + F22*s2**2 + F66*t12**2 + 2*F12*s1*s2
        margin = 1.0 / np.sqrt(max(FI, 1e-9))
        return FI, margin

    def evaluate_laminate_failure(self, Nx, Ny, Nxy):
        """Avalia a Primeira Falha da Lâmina (FPF) em todo o laminado."""
        stresses = self.evaluate_ply_stresses(Nx, Ny, Nxy)
        max_FI = 0.0
        min_margin = 1e6
        critical_ply = 0
        
        for k, (s1, s2, t12) in enumerate(stresses):
            FI, margin = self.tsai_wu(s1, s2, t12)
            if FI > max_FI:
                max_FI = FI
                min_margin = margin
                critical_ply = k + 1 # 1-indexed
                
        return {
            "max_FI": max_FI,
            "min_margin": min_margin,
            "critical_ply": critical_ply,
            "is_safe": max_FI < 1.0
        }
