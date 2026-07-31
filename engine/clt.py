import numpy as np

class Lamina:
    """Representa uma lâmina ortotrópica unidirecional de material compósito."""
    def __init__(self, E1, E2, G12, nu12, XT=2100e6, XC=1400e6, YT=50e6, YC=200e6, S=90e6, t=0.125e-3, name="Carbono/Epóxi"):
        self.E1 = float(E1)
        self.E2 = float(E2)
        self.G12 = float(G12)
        self.nu12 = float(nu12)
        self.nu21 = self.nu12 * self.E2 / self.E1
        
        self.XT = float(XT)
        self.XC = float(XC)
        self.YT = float(YT)
        self.YC = float(YC)
        self.S = float(S)
        self.t = float(t)
        self.name = name
        
        # Matriz de Rigidez Reduzida Q (nos eixos da fibra 1-2)
        denom = 1.0 - self.nu12 * self.nu21
        self.Q = np.array([
            [self.E1 / denom, self.nu12 * self.E2 / denom, 0.0],
            [self.nu12 * self.E2 / denom, self.E2 / denom, 0.0],
            [0.0, 0.0, self.G12]
        ])

    def get_Qbar(self, theta_deg):
        """Calcula a matriz de rigidez transformada Qbar para um ângulo theta (graus)."""
        rad = np.radians(theta_deg)
        m = np.cos(rad)
        n = np.sin(rad)
        
        Q11 = self.Q[0, 0]
        Q22 = self.Q[1, 1]
        Q12 = self.Q[0, 1]
        Q66 = self.Q[2, 2]
        
        Q11_b = Q11*m**4 + 2*(Q12 + 2*Q66)*m**2*n**2 + Q22*n**4
        Q22_b = Q11*n**4 + 2*(Q12 + 2*Q66)*m**2*n**2 + Q22*m**4
        Q12_b = (Q11 + Q22 - 4*Q66)*m**2*n**2 + Q12*(m**4 + n**4)
        Q66_b = (Q11 + Q22 - 2*Q12 - 2*Q66)*m**2*n**2 + Q66*(m**4 + n**4)
        Q16_b = (Q11 - Q12 - 2*Q66)*m**3*n - (Q22 - Q12 - 2*Q66)*m*n**3
        Q26_b = (Q11 - Q12 - 2*Q66)*m*n**3 - (Q22 - Q12 - 2*Q66)*m**3*n
        
        return np.array([
            [Q11_b, Q12_b, Q16_b],
            [Q12_b, Q22_b, Q26_b],
            [Q16_b, Q26_b, Q66_b]
        ])


class Laminate:
    """Representa o compósito laminado e calcula a Teoria Clássica de Laminados (CLT)."""
    def __init__(self, lamina: Lamina, angles):
        self.lamina = lamina
        self.angles = np.array(angles, dtype=float)
        self.N = len(self.angles)
        self.h = self.N * lamina.t
        
        # Coordenadas z de cada interface de camada
        self.z = np.linspace(-self.h / 2.0, self.h / 2.0, self.N + 1)
        
        self.A = np.zeros((3, 3))
        self.B = np.zeros((3, 3))
        self.D = np.zeros((3, 3))
        self.Qbars = []
        
        for k in range(self.N):
            Qbar = lamina.get_Qbar(self.angles[k])
            self.Qbars.append(Qbar)
            zk = self.z[k + 1]
            zk_1 = self.z[k]
            
            self.A += Qbar * (zk - zk_1)
            self.B += 0.5 * Qbar * (zk**2 - zk_1**2)
            self.D += (1.0 / 3.0) * Qbar * (zk**3 - zk_1**3)
            
        # Matriz de Complacência Extensional a = A^-1
        self.a = np.linalg.inv(self.A)
        self.d = np.linalg.inv(self.D)
        
        # Constantes de Engenharia no plano (Equivalentes)
        self.Ex = 1.0 / (self.h * self.a[0, 0])
        self.Ey = 1.0 / (self.h * self.a[1, 1])
        self.Gxy = 1.0 / (self.h * self.a[2, 2])
        self.nu_xy = -self.a[0, 1] / self.a[0, 0]
        self.nu_yx = -self.a[0, 1] / self.a[1, 1]
        
    def get_polar_stiffness(self, n_points=180):
        """Retorna os ângulos (deg) e o módulo Ex(theta) para o gráfico polar de rigidez."""
        theta_degrees = np.linspace(0, 360, n_points)
        Ex_polar = []
        
        for alpha in theta_degrees:
            # Rotaciona a sequência de ângulos por alpha
            rotated_angles = self.angles - alpha
            A_rot = np.zeros((3, 3))
            for angle in rotated_angles:
                A_rot += self.lamina.get_Qbar(angle) * self.lamina.t
            a_rot = np.linalg.inv(A_rot)
            Ex_rot = 1.0 / (self.h * a_rot[0, 0])
            Ex_polar.append(Ex_rot / 1e9) # Em GPa
            
        return theta_degrees, np.array(Ex_polar)
