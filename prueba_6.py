import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


if not hasattr(np, "disp"):
        np.disp = lambda x: print(x)
import roboticstoolbox as rtb
from spatialmath import SE3

# ------------------------- Configurables -------------------------
SIGMA_THRESH = 1e-2
LAMBDA0 = 1e-6
LAMBDA_MAX = 1.0
SIGMA_WARN = 1e-3
ALPHA_IK = 0.6
MAX_DELTA_RAD = 0.1
USE_FULL_JACOBIAN_IF_RTB = True

# ------------------------- Robot con DH -------------------------
robot = rtb.DHRobot(
    [
        rtb.RevoluteDH(alpha=-np.pi/2, a=0.07, d=0.352),
        rtb.RevoluteDH(a=0.36, offset=-np.pi/2),
        rtb.RevoluteDH(alpha=np.pi/2, offset=np.pi),
        rtb.RevoluteDH(d=0.38, alpha=-np.pi/2),
        rtb.RevoluteDH(alpha=np.pi/2),
        rtb.RevoluteDH(d=0.065)
    ], name="IRB140"
)

DOF = robot.n

# ------------------------- Límites articulares -------------------------
#Agregar un botón para desactivar los limites
# Ver si indicarle al usuario que en la configuración que el quiso hacer el movimiento se puede o no hacer el
# recorrido de la trayectoria recta que se pidió en función de los limites articulres y del alcance del robot
#  
JOINT_LIMITS = [
    (np.deg2rad(-180.0), np.deg2rad(180.0)),   # q1
    (np.deg2rad(-100.0), np.deg2rad(100.0)),   # q2
    (np.deg2rad(-220.0), np.deg2rad(60.0)),    # q3
    (np.deg2rad(-200.0), np.deg2rad(200.0)),   # q4
    (np.deg2rad(-120.0), np.deg2rad(120.0)),   # q5
    (np.deg2rad(-400.0), np.deg2rad(400.0)),   # q6
]

def clip_joints(q):
    """Asegura que q esté dentro de los límites articulares."""
    q_clipped = q.copy()
    for i, (low, high) in enumerate(JOINT_LIMITS):
        q_clipped[i] = np.clip(q[i], low, high)
    return q_clipped

# ------------------------- FK y Jacobiano -------------------------
def forward_kinematics(joint_values):
    T = robot.fkine(joint_values)
    Ts_se3 = robot.fkine_all(joint_values)
    Ts = [T.A for T in Ts_se3]
    return T.A, Ts

def numeric_jacobian(joint_values, eps=1e-8):
    J6 = robot.jacob0(joint_values)
    Jv = np.asarray(J6)[0:3, :]
    return Jv

def numeric_jacobian_full6(joint_values, eps=1e-8):
    J6 = robot.jacob0(joint_values)
    return np.asarray(J6)

# ------------------------- (resto igual) -------------------------

def manipulability_ellipsoid_from_J(Jv):
    M = Jv @ Jv.T
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    axes = np.sqrt(np.clip(eigvals, 0, None))
    try:
        svals = np.linalg.svd(Jv, compute_uv=False)
        # Computar tambien las matrices U y V que  son usados  con los sigmas.
    except Exception:
        svals = np.array([])
    cond = np.inf
    if svals.size > 0 and svals[-1] > 0:
        cond = svals[0] / svals[-1]
    return axes, eigvecs, M, svals, cond

def compute_lambda_from_sigma(sigma_min, sigma_thresh=SIGMA_THRESH, lambda0=LAMBDA0, lambda_max=LAMBDA_MAX, eps=1e-12):
    sigma_min_clamped = max(sigma_min, eps)
    if sigma_min_clamped >= sigma_thresh:
        return lambda0
    ratio = (sigma_thresh / sigma_min_clamped)
    lam = lambda0 * (ratio ** 2)
    if lam > lambda_max:
        lam = lambda_max
    return lam

def damped_pinv_adaptive(Jv, sigma_thresh=SIGMA_THRESH, lambda0=LAMBDA0, lambda_max=LAMBDA_MAX):
    try:
        svals = np.linalg.svd(Jv, compute_uv=False)
    except Exception:
        svals = np.array([])
    sigma_min = svals[-1] if svals.size > 0 else 0.0
    lam = compute_lambda_from_sigma(sigma_min, sigma_thresh=sigma_thresh, lambda0=lambda0, lambda_max=lambda_max)
    JJ = Jv @ Jv.T + (lam ** 2) * np.eye(Jv.shape[0])
    try:
        invJJ = np.linalg.inv(JJ)
    except np.linalg.LinAlgError:
        invJJ = np.linalg.pinv(JJ)
    Jpinv = Jv.T @ invJJ
    return Jpinv, lam, svals

# ------------------------- Plotting -------------------------
def plot_robot_and_ellipsoid(ax, joint_values, highlight_singular=False, planned_path_pts=None, planned_markers=None):
    ax.cla()
    ax.set_box_aspect([1,1,1])
    ax.grid(True)
    lim = 0.9
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, 1.2)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    Ttcp, Ts = forward_kinematics(joint_values)
    origins = np.array([T[:3,3] for T in Ts])
    ax.plot(origins[:,0], origins[:,1], origins[:,2], '-o', linewidth=2, markersize=4)
    for i in range(len(origins)-1):
        p0 = origins[i]; p1 = origins[i+1]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], linewidth=4)

    Jv = numeric_jacobian(joint_values)
    axes_len, axes_dir, M, svals, cond = manipulability_ellipsoid_from_J(Jv)

    p = Ttcp[:3,3]
    u = np.linspace(0, 2*np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    X = np.outer(np.cos(u), np.sin(v))
    Y = np.outer(np.sin(u), np.sin(v))
    Z = np.outer(np.ones_like(u), np.cos(v))
    ell = np.zeros((X.shape[0], X.shape[1], 3))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            vec = np.array([X[i,j], Y[i,j], Z[i,j]])
            point = (axes_len[0]*vec[0]*axes_dir[:,0] +
                     axes_len[1]*vec[1]*axes_dir[:,1] +
                     axes_len[2]*vec[2]*axes_dir[:,2])
            ell[i,j,:] = p + point

    alpha = 0.35
    ax.plot_surface(ell[:,:,0], ell[:,:,1], ell[:,:,2], rstride=1, cstride=1, alpha=alpha)

    for k in range(3):
        start = p
        end = p + axes_len[k] * axes_dir[:,k]
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], linewidth=3)

    if planned_path_pts is not None and len(planned_path_pts) > 0:
        pts = np.array(planned_path_pts)
        ax.plot(pts[:,0], pts[:,1], pts[:,2], '--', linewidth=2)
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=10)

    if planned_markers is not None:
        for lab, pt in planned_markers:
            ax.scatter([pt[0]], [pt[1]], [pt[2]], s=60)
            ax.text(pt[0], pt[1], pt[2], f' {lab}', fontsize=10)

    return cond, svals

# ------------------------- IK incremental -------------------------
def jacobian_ik_step_adaptive(q_init, p_desired,
                              alpha=ALPHA_IK,
                              sigma_thresh=SIGMA_THRESH,
                              lambda0=LAMBDA0,
                              lambda_max=LAMBDA_MAX,
                              max_delta=MAX_DELTA_RAD,
                              use_full_jacobian=False):
    q = q_init.copy()
    T, _ = forward_kinematics(q)
    p = T[:3,3]
    e = p_desired - p
    if np.linalg.norm(e) < 1e-6:
        return q, 0.0, np.array([]), 0.0, None

    Jv = numeric_jacobian(q)
    Jpinv, lam, svals = damped_pinv_adaptive(Jv, sigma_thresh=sigma_thresh, lambda0=lambda0, lambda_max=lambda_max)

    q_dot = Jpinv @ e
    norm_qdot = np.linalg.norm(q_dot)
    if norm_qdot > max_delta:
        q_dot = q_dot / norm_qdot * max_delta
    q = q + alpha * q_dot

    sigma_min_full = None
    cond_full = None
    if use_full_jacobian:
        try:
            Jfull = numeric_jacobian_full6(q)
            svals_full = np.linalg.svd(Jfull, compute_uv=False)
            if svals_full.size > 0:
                sigma_min_full = svals_full[-1]
                cond_full = svals_full[0] / max(svals_full[-1], 1e-12)
        except Exception:
            sigma_min_full = None
            cond_full = None

    sigma_min_pos = svals[-1] if svals.size>0 else 0.0
    return q, lam, svals, (sigma_min_full if sigma_min_full is not None else sigma_min_pos), cond_full

def generate_linear_trajectory(q_start, p_target, steps=200, use_full_jacobian=False):
    T0, _ = forward_kinematics(q_start)
    p0 = T0[:3,3]
    q = q_start.copy()
    traj = []
    for t in np.linspace(0, 1, steps):
        pd = (1.0 - t) * p0 + t * p_target
        q, lam, svals, sigma_min_maybe_full, cond_full = jacobian_ik_step_adaptive(
            q, pd,
            alpha=ALPHA_IK,
            sigma_thresh=SIGMA_THRESH,
            lambda0=LAMBDA0,
            lambda_max=LAMBDA_MAX,
            max_delta=MAX_DELTA_RAD,
            use_full_jacobian=use_full_jacobian
        )
        s_min_pos = svals[-1] if svals.size>0 else 0.0
        sigma_min_full = sigma_min_maybe_full if (use_full_jacobian and sigma_min_maybe_full is not None) else (s_min_pos)
        traj.append((q.copy(), lam, s_min_pos, sigma_min_full, cond_full))
    return traj

def solve_ik_to_point(q_init, p_target, max_iters=800, tol=1e-5, use_full_jacobian=False):
    q = q_init.copy()
    for i in range(max_iters):
        T, _ = forward_kinematics(q)
        p = T[:3,3]
        err = p_target - p
        if np.linalg.norm(err) < tol:
            return q, True, i
        q, lam, svals, _, _ = jacobian_ik_step_adaptive(q, p_target, use_full_jacobian=use_full_jacobian)
    return q, False, max_iters

# ------------------------- NUEVO: helper para q0 según config -------------------------
def q0_from_config(config):
    # config = [s,e,w], cada uno 0 o 1
    s, e, w = config
    q0 = np.zeros(6)
    q0[0] = -np.pi/4 if s == 0 else np.pi/4
    q0[2] = -np.pi/4 if e == 0 else np.pi/4
    q0[4] = 0 if w == 0 else np.pi/2
    return q0

# ------------------------- GUI -------------------------
class ManipulabilityApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('IRB140 - Elipsoide de manipulabilidad (traslacional) - DLS adaptativo (con origen/destino)')
        self.joints = np.zeros(DOF)
        self.playing = False
        self.traj = []
        self.traj_idx = 0
        self.use_full_jacobian_monitor = USE_FULL_JACOBIAN_IF_RTB and (robot is not None)
        self.planned_pts = None
        self.planned_markers = None
        self.active_config = None  # NUEVO: guarda config activa

        layout = QtWidgets.QHBoxLayout(self)
        ctrl = QtWidgets.QVBoxLayout()
        self.sliders = []
        self.labels = []
        for i in range(DOF):
            lbl = QtWidgets.QLabel(f'Joint {i+1}: 0.000 rad')
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(-180)
            slider.setMaximum(180)
            slider.setValue(0)
            slider.setSingleStep(1)
            slider.valueChanged.connect(self.make_slider_callback(i, lbl))
            self.sliders.append(slider)
            self.labels.append(lbl)
            ctrl.addWidget(lbl)
            ctrl.addWidget(slider)

        self.traj_btn = QtWidgets.QPushButton('Generar trayectoria rectilínea (origen->destino)')
        self.traj_btn.clicked.connect(self.generate_trajectory_with_origin_dest)
        ctrl.addWidget(self.traj_btn)

        hplay = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton('Play traj')
        self.play_btn.clicked.connect(self.play_pause)
        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop)
        hplay.addWidget(self.play_btn)
        hplay.addWidget(self.stop_btn)
        ctrl.addLayout(hplay)


        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)  # valor central = velocidad normal
        self.speed_slider.valueChanged.connect(self.change_speed)
        ctrl.addWidget(QtWidgets.QLabel("Velocidad animación"))
        ctrl.addWidget(self.speed_slider)
        self.speed_factor = 1.0


        self.info_label = QtWidgets.QLabel('Condición: -')
        ctrl.addWidget(self.info_label)

        if robot is not None:
            self.full_jac_chk = QtWidgets.QCheckBox('Monitor jacobiana completa 6xN (RTB)')
            self.full_jac_chk.setChecked(self.use_full_jacobian_monitor)
            self.full_jac_chk.stateChanged.connect(self.toggle_full_jac)
            ctrl.addWidget(self.full_jac_chk)

        ctrl.addStretch(1)
        layout.addLayout(ctrl, 1)

        self.fig = Figure(figsize=(6,6))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas, 3)

        cond, svals = plot_robot_and_ellipsoid(self.ax, self.joints)
        s_min = svals[-1] if svals.size>0 else 0.0
        self.info_label.setText(self._info_text(cond, s_min, 0.0, None, None))
        self.canvas.draw()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(20)
        self.timer.timeout.connect(self._play_step)

    def toggle_full_jac(self, state):
        self.use_full_jacobian_monitor = (state == QtCore.Qt.Checked)

    def make_slider_callback(self, idx, label):
        def cb(val):
            angle = np.deg2rad(val)
            label.setText(f'Joint {idx+1}: {angle:.3f} rad')
            self.joints[idx] = angle
            self.joints = clip_joints(self.joints)
            cond, svals = plot_robot_and_ellipsoid(self.ax, self.joints)
            s_min = svals[-1] if svals.size > 0 else 0.0
            self.info_label.setText(self._info_text(cond, s_min, 0.0, None, None))
            self.canvas.draw()
        return cb

    def _info_text(self, cond_pos, s_min_pos, lam, sigma_min, cond_full=None):
        text = f's_min_pos={s_min_pos:.3e} | λ={lam:.3e}'
        if sigma_min is not None:
            text += f' | s_min={sigma_min:.3e}'
        if cond_pos is not None:
            text += f' | cond_pos={cond_pos:.3e}'
        if cond_full is not None:
            text += f' | cond_full={cond_full:.3e}'
        return text

    def change_speed(self, val):
        # val va de 0 a 100
        t = val / 100.0  # normalizar
        # mapear t∈[0,1] a factor ∈ [1/3, 5]
        self.speed_factor = (1/3) * (1 - t) + 5 * t
        base_interval = 20  # ms
        self.timer.setInterval(int(base_interval / self.speed_factor))


    def generate_trajectory_with_origin_dest(self):
        dlg = QtWidgets.QInputDialog(self)
        dlg.setLabelText(
            'Ingrese ORIGEN x,y,z ; DESTINO x,y,z ; CONFIG (opcional [s,e,w])\n'
            'Ejemplo sin config: 0.2,0,0 ; 0.5,0.1,0.2\n'
            'Ejemplo con config: 0.2,0,0 ; 0.5,0.1,0.2 ; 1,0,1'
        )
        ok = dlg.exec_()
        if not ok:
            return
        text = dlg.textValue()
        try:
            parts = [p.strip() for p in text.split(';')]
            if len(parts) == 1:
                origin_text, dest_text, config_text = '', parts[0], None
            elif len(parts) == 2:
                origin_text, dest_text, config_text = parts[0], parts[1], None
            else:
                origin_text, dest_text, config_text = parts[0], parts[1], parts[2]

            if origin_text == '':
                T0, _ = forward_kinematics(self.joints)
                p0 = T0[:3,3]
            else:
                p0 = np.array([float(x.strip()) for x in origin_text.split(',')])
            pd = np.array([float(x.strip()) for x in dest_text.split(',')])

            config = None
            if config_text is not None:
                config = np.array([int(x.strip()) for x in config_text.split(',')])
        except Exception:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Formato inválido. Use: x,y,z ; x,y,z ; config(opcional [s,e,w])')
            return

        Tcurr, _ = forward_kinematics(self.joints)
        p_tcp = Tcurr[:3,3]
        q_start = self.joints.copy()
        joint_steps = []
        move_to_origin_success = True

        if config is not None:
            q0 = q0_from_config(config)
            q_origin, ok_conv, its = solve_ik_to_point(q0, p0, max_iters=800, tol=1e-5, use_full_jacobian=self.use_full_jacobian_monitor)
            self.active_config = config
        else:
            if np.linalg.norm(p0 - p_tcp) > 1e-6:
                q_origin, ok_conv, its = solve_ik_to_point(q_start, p0, max_iters=800, tol=1e-5, use_full_jacobian=self.use_full_jacobian_monitor)
            else:
                q_origin = q_start.copy()
            self.active_config = None

        steps_js = 120
        for t in np.linspace(0, 1, steps_js):
            q_interp = (1.0 - t) * q_start + t * q_origin
            Jv = numeric_jacobian(q_interp)
            try:
                svals = np.linalg.svd(Jv, compute_uv=False)
            except Exception:
                svals = np.array([])
            s_min_pos = svals[-1] if svals.size>0 else 0.0
            lam = compute_lambda_from_sigma(s_min_pos)
            joint_steps.append((q_interp.copy(), lam, s_min_pos, s_min_pos, None))

        q_for_cartesian = q_origin.copy()
        steps_cart = 200
        traj_cart = generate_linear_trajectory(q_for_cartesian, pd, steps=steps_cart, use_full_jacobian=self.use_full_jacobian_monitor)

        full_traj = joint_steps + traj_cart
        if len(full_traj) == 0:
            QtWidgets.QMessageBox.information(self, 'Trayectoria', 'No se generó trayectoria.')
            return

        self.traj = full_traj
        self.traj_idx = 0
        planned_pts = [ (1.0 - t) * p0 + t * pd for t in np.linspace(0,1,60) ]
        planned_markers = [('Origen', p0), ('Destino', pd)]

        cond, svals = plot_robot_and_ellipsoid(self.ax, self.joints, planned_path_pts=planned_pts, planned_markers=planned_markers)
        self.canvas.draw()
        QtWidgets.QMessageBox.information(self, 'Trayectoria', f'Trayectoria generada: {len(full_traj)} pasos.')
        self.planned_pts = planned_pts
        self.planned_markers = planned_markers

    def _play_step(self):
        if not self.traj or self.traj_idx >= len(self.traj):
            self.timer.stop()
            self.playing = False
            self.play_btn.setText('Play traj')
            for s in self.sliders:
                s.setEnabled(True)
            self.active_config = None
            return

        q, lam, s_min_pos, s_min_full, cond_full = self.traj[self.traj_idx]
        self.traj_idx += 1
        self.joints = clip_joints(q)
        for s in self.sliders:
            s.setEnabled(False)
        for i, s in enumerate(self.sliders):
            s.blockSignals(True)
            s.setValue(int(np.rad2deg(self.joints[i])))
            self.labels[i].setText(f'Joint {i+1}: {self.joints[i]:.3f} rad')
            s.blockSignals(False)
        cond_pos, _ = plot_robot_and_ellipsoid(self.ax, self.joints, planned_path_pts=self.planned_pts, planned_markers=self.planned_markers)
        self.info_label.setText(self._info_text(cond_pos, s_min_pos, lam, s_min_full, cond_full))
        self.canvas.draw()

    def play_pause(self):
        if not self.traj:
            QtWidgets.QMessageBox.information(self, 'Info', 'No hay trayectoria generada.')
            return
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_btn.setText('Play traj')
        else:
            self.timer.start()
            self.playing = True
            self.play_btn.setText('Pause')

    def stop(self):
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.play_btn.setText('Play traj')
        self.traj_idx = 0
        for s in self.sliders:
            s.setEnabled(True)
        self.active_config = None

# ------------------------- Main -------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = ManipulabilityApp()
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()