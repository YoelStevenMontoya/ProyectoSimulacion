"""
Proyecto corregido: Flujo incompresible 2D con ψ-ω, Newton-Raphson y Jacobiano numérico.
Malla reducida (25x9) para hacer el cálculo factible.
Corregidos errores de plotting y se añadió line search.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

# ============================================================
# 1. PARÁMETROS FÍSICOS Y NUMÉRICOS (malla aún más pequeña)
# ============================================================
Lx = 24.0           # Longitud del canal (múltiplo de dx para ajustar obstáculos)
Ly = 8.0            # Altura
dx = 1.0
dy = 1.0
nx = int(Lx / dx) + 1   # 25
ny = int(Ly / dy) + 1   # 9

nu = 0.2            # Viscosidad (más alta para estabilidad)
U_inlet = 1.0
tol = 1e-5
max_iter = 40
eps_jac = 1e-7
alpha_init = 1.0    # paso inicial para line search

# ============================================================
# 2. MALLA Y OBSTÁCULOS (ajustados al nuevo tamaño)
# ============================================================
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y, indexing='ij')  # forma (nx, ny)

fluid = np.ones((nx, ny), dtype=bool)

# Obstáculo 1: esquina superior izquierda (tamaño 6 x 2)
ox1_start, ox1_end = 0, 6
oy1_start, oy1_end = Ly - 2, Ly
fluid[int(ox1_start/dx):int(ox1_end/dx)+1, int(oy1_start/dy):int(oy1_end/dy)+1] = False

# Obstáculo 2: pegado al suelo, de ancho 2, alto 4, empezando en x=16
ox2_start, ox2_end = 16, 18
oy2_start, oy2_end = 0, 4
fluid[int(ox2_start/dx):int(ox2_end/dx)+1, int(oy2_start/dy):int(oy2_end/dy)+1] = False

# ============================================================
# 3. INICIALIZACIÓN MEJORADA (flujo potencial aproximado)
# ============================================================
psi = np.zeros((nx, ny))
omega = np.zeros((nx, ny))

# Perfil lineal en la entrada
for j in range(ny):
    psi[0, j] = U_inlet * y[j]

# Inicializar el interior con un flujo uniforme (sin obstáculos)
for i in range(1, nx):
    psi[i, :] = psi[0, :]   # flujo uniforme

# Ajustar obstáculos
psi[~fluid] = 0.0
omega[~fluid] = 0.0

# ============================================================
# 4. FUNCIONES AUXILIARES (actualizadas)
# ============================================================
def apply_bc(psi, omega):
    """Aplica condiciones de contorno a psi y omega."""
    # psi
    for j in range(ny):
        psi[0, j] = U_inlet * y[j]
    psi[-1, :] = psi[-2, :]          # Neumann salida
    psi[:, 0] = 0.0
    psi[:, -1] = U_inlet * Ly
    psi[~fluid] = 0.0

    # omega
    # Pared inferior
    for i in range(1, nx-1):
        if fluid[i, 1]:
            omega[i, 0] = -2.0 * psi[i, 1] / (dy*dy)
        else:
            omega[i, 0] = 0.0
    # Pared superior
    for i in range(1, nx-1):
        if fluid[i, ny-2]:
            omega[i, ny-1] = -2.0 * (psi[i, ny-2] - U_inlet*Ly) / (dy*dy)
        else:
            omega[i, ny-1] = 0.0
    omega[0, :] = 0.0
    omega[-1, :] = omega[-2, :]
    omega[~fluid] = 0.0

def compute_velocities(psi):
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2*dy)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2*dx)
    u[:, 0] = (psi[:, 1] - psi[:, 0]) / dy
    u[:, -1] = (psi[:, -1] - psi[:, -2]) / dy
    v[0, :] = -(psi[1, :] - psi[0, :]) / dx
    v[-1, :] = -(psi[-1, :] - psi[-2, :]) / dx
    u[~fluid] = 0.0
    v[~fluid] = 0.0
    return u, v

def residual(psi, omega):
    """Residual del sistema acoplado."""
    nx, ny = psi.shape
    u, v = compute_velocities(psi)
    R = np.zeros(2 * nx * ny)
    idx = lambda i, j, var: 2*(i*ny + j) + var

    for i in range(nx):
        for j in range(ny):
            # --- Ecuación para psi ---
            if fluid[i, j] and (1 <= i <= nx-2) and (1 <= j <= ny-2):
                lap = (psi[i+1,j]-2*psi[i,j]+psi[i-1,j])/dx**2 + \
                      (psi[i,j+1]-2*psi[i,j]+psi[i,j-1])/dy**2
                R[idx(i,j,0)] = lap + omega[i,j]
            else:
                # Dirichlet para psi
                target = 0.0
                if i == 0:
                    target = U_inlet * y[j]
                elif i == nx-1:
                    target = psi[nx-2, j]
                elif j == 0:
                    target = 0.0
                elif j == ny-1:
                    target = U_inlet * Ly
                R[idx(i,j,0)] = psi[i,j] - target

            # --- Ecuación para omega ---
            if fluid[i,j] and (1 <= i <= nx-2) and (1 <= j <= ny-2):
                # Convección upwind
                if u[i,j] > 0:
                    dwdx = (omega[i,j] - omega[i-1,j])/dx
                else:
                    dwdx = (omega[i+1,j] - omega[i,j])/dx
                if v[i,j] > 0:
                    dwdy = (omega[i,j] - omega[i,j-1])/dy
                else:
                    dwdy = (omega[i,j+1] - omega[i,j])/dy
                conv = u[i,j]*dwdx + v[i,j]*dwdy
                lap_omega = (omega[i+1,j]-2*omega[i,j]+omega[i-1,j])/dx**2 + \
                            (omega[i,j+1]-2*omega[i,j]+omega[i,j-1])/dy**2
                R[idx(i,j,1)] = conv - nu * lap_omega
            else:
                # Dirichlet / Neumann para omega
                target = 0.0
                if i == 0:
                    target = 0.0
                elif i == nx-1:
                    target = omega[nx-2, j]
                elif j == 0:
                    if fluid[i,1]:
                        target = -2.0 * psi[i,1] / dy**2
                elif j == ny-1:
                    if fluid[i,ny-2]:
                        target = -2.0 * (psi[i,ny-2] - U_inlet*Ly) / dy**2
                R[idx(i,j,1)] = omega[i,j] - target
    return R

def jacobian_numerical(psi, omega, eps=1e-7):
    """Jacobiano numérico por diferencias finitas."""
    nx, ny = psi.shape
    n_vars = 2 * nx * ny
    R0 = residual(psi, omega)
    J = np.zeros((n_vars, n_vars))
    psi_pert = psi.copy()
    omega_pert = omega.copy()

    for var in range(n_vars):
        node = var // 2
        i = node // ny
        j = node % ny
        comp = var % 2
        if comp == 0:
            orig = psi_pert[i,j]
            psi_pert[i,j] += eps
        else:
            orig = omega_pert[i,j]
            omega_pert[i,j] += eps

        R_pert = residual(psi_pert, omega_pert)
        J[:, var] = (R_pert - R0) / eps

        if comp == 0:
            psi_pert[i,j] = orig
        else:
            omega_pert[i,j] = orig
    return J

# ============================================================
# 5. NEWTON-RAPHSON CON LINE SEARCH
# ============================================================
apply_bc(psi, omega)
print("Iniciando Newton-Raphson con line search...")
print(f"Malla: {nx} x {ny} = {nx*ny} nodos → {2*nx*ny} incógnitas")

residual_norm = []
for it in range(max_iter):
    R = residual(psi, omega)
    normR = np.linalg.norm(R)
    residual_norm.append(normR)
    print(f"Iter {it+1:2d}: ||R|| = {normR:.3e}")

    if normR < tol:
        print("Convergencia alcanzada.")
        break

    # Jacobiano
    print("   Calculando Jacobiano numérico...")
    J = jacobian_numerical(psi, omega, eps=eps_jac)

    # Resolver sistema
    print("   Resolviendo sistema lineal...")
    delta = solve(J, -R)

    # Line search: reducir paso si empeora el residual
    alpha = alpha_init
    psi_try = psi.copy()
    omega_try = omega.copy()
    for _ in range(8):  # máximo 8 reducciones
        # Actualizar prueba
        for var in range(2*nx*ny):
            node = var // 2
            i = node // ny
            j = node % ny
            comp = var % 2
            if comp == 0:
                psi_try[i,j] = psi[i,j] + alpha * delta[var]
            else:
                omega_try[i,j] = omega[i,j] + alpha * delta[var]
        apply_bc(psi_try, omega_try)
        R_try = residual(psi_try, omega_try)
        norm_try = np.linalg.norm(R_try)
        if norm_try < normR:
            # Aceptamos el paso
            psi, omega = psi_try, omega_try
            print(f"   Paso aceptado con alpha = {alpha:.3f} (||R|| nuevo = {norm_try:.3e})")
            break
        else:
            alpha *= 0.5
            psi_try = psi.copy()
            omega_try = omega.copy()
    else:
        print("   No se encontró paso que reduzca el residual. Se detiene.")
        break

print("\nSimulación finalizada.")

# ============================================================
# 6. VISUALIZACIÓN (corregida)
# ============================================================
u_final, v_final = compute_velocities(psi)

plt.figure(figsize=(14,10))

# Para graficar correctamente: contourf espera X, Y con la misma forma que psi.
# Como X e Y tienen forma (nx, ny) y psi también, usamos psi directamente (sin transponer).
plt.subplot(2,2,1)
cf1 = plt.contourf(X, Y, psi, levels=30, cmap='jet')
plt.colorbar(cf1)
plt.title("Función corriente ψ")
plt.xlabel("x"); plt.ylabel("y")

plt.subplot(2,2,2)
cf2 = plt.contourf(X, Y, omega, levels=30, cmap='coolwarm')
plt.colorbar(cf2)
plt.title("Vorticidad ω")
plt.xlabel("x"); plt.ylabel("y")

# Campo de velocidades (cada 2 puntos)
plt.subplot(2,2,3)
skip = 2
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip],
           u_final[::skip, ::skip], v_final[::skip, ::skip], scale=25)
plt.title("Campo de velocidades")
plt.xlabel("x"); plt.ylabel("y")

# Líneas de corriente
plt.subplot(2,2,4)
plt.contour(X, Y, psi, levels=30, colors='k', linewidths=0.8)
plt.title("Líneas de corriente")
plt.xlabel("x"); plt.ylabel("y")

# Dibujar obstáculos en todas las subfiguras
for ax in [plt.subplot(2,2,1), plt.subplot(2,2,2), plt.subplot(2,2,3), plt.subplot(2,2,4)]:
    # Obstáculo 1
    ax.fill_betweenx([oy1_start, oy1_end], ox1_start, ox1_end, color='gray', alpha=0.6)
    # Obstáculo 2
    ax.fill_betweenx([oy2_start, oy2_end], ox2_start, ox2_end, color='gray', alpha=0.6)
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)

plt.tight_layout()
plt.show()

# Gráfica de convergencia
plt.figure()
plt.semilogy(residual_norm, 'o-')
plt.xlabel("Iteración de Newton")
plt.ylabel("Norma del residual")
plt.title("Convergencia de Newton-Raphson")
plt.grid(True)
plt.show()