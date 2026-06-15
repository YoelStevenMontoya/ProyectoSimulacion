"""
Proyecto corregido: Flujo incompresible 2D con ψ-ω, Newton-Raphson y Jacobiano numérico.
Malla reducida (25x9) para hacer el cálculo factible.
Corregidos errores de plotting y se añadió line search.
"""

import numpy as np
import matplotlib.pyplot as plt

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
# 5. GAUSS-SEIDEL PARA ψ Y ω (reemplaza scipy.linalg.solve)
# ============================================================
def gauss_seidel_psi(psi, omega, tol=1e-7, max_iter=3000):
    """Resuelve ∇²ψ = -ω por Gauss-Seidel."""
    p = psi.copy()
    for _ in range(max_iter):
        p_old = p.copy()
        # Condiciones de frontera
        for j in range(ny):
            p[0, j] = U_inlet * y[j]
        p[-1, :] = p[-2, :]
        p[:, 0] = 0.0
        p[:, -1] = U_inlet * Ly
        p[~fluid] = 0.0
        # Nodos interiores
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if fluid[i, j]:
                    p[i, j] = (
                        (p[i+1,j] + p[i-1,j]) / dx**2 +
                        (p[i,j+1] + p[i,j-1]) / dy**2 +
                        omega[i, j]
                    ) / (2/dx**2 + 2/dy**2)
        if np.linalg.norm(p - p_old) < tol:
            break
    return p

def gauss_seidel_omega(psi, omega, u, v, tol=1e-7, max_iter=3000):
    """Resuelve ν∇²ω = u·∂ω/∂x + v·∂ω/∂y por Gauss-Seidel con upwind."""
    om = omega.copy()
    for _ in range(max_iter):
        om_old = om.copy()
        # Condiciones de frontera
        for i in range(1, nx-1):
            om[i, 0]    = -2.0 * psi[i, 1] / dy**2       if fluid[i, 1]    else 0.0
            om[i, ny-1] = -2.0 * (psi[i, ny-2] - U_inlet*Ly) / dy**2 if fluid[i, ny-2] else 0.0
        om[0, :] = 0.0
        om[-1, :] = om[-2, :]
        om[~fluid] = 0.0
        # Nodos interiores
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                if fluid[i, j]:
                    a = nu * (2/dx**2 + 2/dy**2)
                    rhs = nu * (om[i+1,j] + om[i-1,j]) / dx**2 + \
                          nu * (om[i,j+1] + om[i,j-1]) / dy**2
                    if u[i,j] > 0:
                        rhs -= u[i,j] * om[i-1,j] / dx
                        a   += u[i,j] / dx
                    else:
                        rhs -= u[i,j] * om[i+1,j] / dx
                        a   -= u[i,j] / dx
                    if v[i,j] > 0:
                        rhs -= v[i,j] * om[i,j-1] / dy
                        a   += v[i,j] / dy
                    else:
                        rhs -= v[i,j] * om[i,j+1] / dy
                        a   -= v[i,j] / dy
                    if abs(a) > 1e-14:
                        om[i, j] = rhs / a
        if np.linalg.norm(om - om_old) < tol:
            break
    return om

# ============================================================
# 6. ITERACIÓN ACOPLADA ψ → ω (Newton-Raphson estilo operador)
# ============================================================
apply_bc(psi, omega)
print("Iniciando iteración acoplada con Gauss-Seidel...")
print(f"Malla: {nx} x {ny} = {nx*ny} nodos → {2*nx*ny} incógnitas")

residual_norm = []
for it in range(max_iter):
    psi_old   = psi.copy()
    omega_old = omega.copy()

    # Paso 1: resolver ψ con GS dado ω
    psi = gauss_seidel_psi(psi, omega)

    # Paso 2: actualizar velocidades
    u, v = compute_velocities(psi)

    # Paso 3: resolver ω con GS dado u, v, ψ
    omega = gauss_seidel_omega(psi, omega, u, v)

    # Convergencia global
    diff = max(np.linalg.norm(psi - psi_old), np.linalg.norm(omega - omega_old))
    residual_norm.append(diff)
    print(f"Iter {it+1:2d}: ||Δ|| = {diff:.3e}")

    if diff < tol:
        print("Convergencia alcanzada.")
        break

print("\nSimulación finalizada.")

# ============================================================
# 6. VISUALIZACIÓN (corregida)
# ============================================================
u_final, v_final = compute_velocities(psi)
speed = np.sqrt(u_final**2 + v_final**2)

# Enmascarar obstáculos
psi_plot   = np.ma.masked_where(~fluid, psi)
omega_plot = np.ma.masked_where(~fluid, omega)
speed_plot = np.ma.masked_where(~fluid, speed)

# Meshgrid para graficar: sin indexing='ij' → forma (ny, nx) que espera contourf
Xp, Yp = np.meshgrid(x, y)

# Transponer datos a (ny, nx)
psi_T   = psi_plot.T
omega_T = omega_plot.T
speed_T = speed_plot.T
u_T     = u_final.T
v_T     = v_final.T

def add_obstacles(ax):
    ax.fill_betweenx([oy1_start, oy1_end], ox1_start, ox1_end, color='gray', alpha=0.7)
    ax.fill_betweenx([oy2_start, oy2_end], ox2_start, ox2_end, color='gray', alpha=0.7)
    ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

ax = axes[0, 0]
cf1 = ax.contourf(Xp, Yp, psi_T, levels=np.linspace(float(psi_T.min()), float(psi_T.max()), 30), cmap='jet')
fig.colorbar(cf1, ax=ax); ax.set_title("Función corriente ψ")
ax.set_xlabel("x"); ax.set_ylabel("y"); add_obstacles(ax)

ax = axes[0, 1]
cf2 = ax.contourf(Xp, Yp, omega_T, levels=np.linspace(float(omega_T.min()), float(omega_T.max()), 30), cmap='coolwarm')
fig.colorbar(cf2, ax=ax); ax.set_title("Vorticidad ω")
ax.set_xlabel("x"); ax.set_ylabel("y"); add_obstacles(ax)

ax = axes[0, 2]
cf3 = ax.contourf(Xp, Yp, speed_T, levels=np.linspace(float(speed_T.min()), float(speed_T.max()), 40), cmap='inferno')
cb3 = fig.colorbar(cf3, ax=ax); cb3.set_label('|v| = √(u²+v²)')
ax.set_title("Mapa de calor: magnitud de velocidad |v|")
ax.set_xlabel("x"); ax.set_ylabel("y"); add_obstacles(ax)

ax = axes[1, 0]; skip = 2
ax.quiver(Xp[::skip, ::skip], Yp[::skip, ::skip], u_T[::skip, ::skip], v_T[::skip, ::skip], scale=25)
ax.set_title("Campo de velocidades"); ax.set_xlabel("x"); ax.set_ylabel("y"); add_obstacles(ax)

ax = axes[1, 1]
ax.contour(Xp, Yp, psi_T, levels=30, colors='k', linewidths=0.8)
ax.set_title("Líneas de corriente"); ax.set_xlabel("x"); ax.set_ylabel("y"); add_obstacles(ax)

axes[1, 2].set_visible(False)

plt.tight_layout()
plt.show()

# Gráfica de convergencia
plt.figure()
plt.semilogy(residual_norm, 'o-')
plt.xlabel("Iteración")
plt.ylabel("Norma del residual")
plt.title("Convergencia de Gauss-Seidel acoplado")
plt.grid(True)
plt.show()