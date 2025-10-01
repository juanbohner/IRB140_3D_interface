# prueba_6_pyqtgraph.py
import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Necesita roboticstoolbox y spatialmath como en tu versión original.
if not hasattr(np, "disp"):
    np.disp = lambda x: print(x)
import roboticstoolbox as rtb
from spatialmath import SE3
from stl import mesh  # de numpy-stl


# ------------------------- Configurables -------------------------
SIGMA_THRESH = 1e-2
LAMBDA0 = 1e-6
LAMBDA_MAX = 1.0
SIGMA_WARN = 1e-3
ALPHA_IK = 0.6
MAX_DELTA_RAD = 0.1
USE_FULL_JACOBIAN_IF_RTB = True

# ------------------------- Robot con DH (igual que en tu archivo) -------------------------
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
JOINT_LIMITS = [
    (np.deg2rad(-180.0), np.deg2rad(180.0)),   # q1
    (np.deg2rad(-100.0), np.deg2rad(100.0)),   # q2
    (np.deg2rad(-220.0), np.deg2rad(60.0)),    # q3
    (np.deg2rad(-200.0), np.deg2rad(200.0)),   # q4
    (np.deg2rad(-120.0), np.deg2rad(120.0)),   # q5
    (np.deg2rad(-400.0), np.deg2rad(400.0)),   # q6
]

def clip_joints(q):
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

# ------------------------- Manipulability (SVD-based) -------------------------
def manipulability_ellipsoid_from_J(Jv):
    """
    Retorna:
      axes_len: array(3,) = longitudes de los ejes (σ_i)
      axes_dir: array(3,3) = columnas = direcciones (U)
      M: matriz J J^T
      svals: valores singulares (σ)
      cond: condicion numérica (σ_max / σ_min)
    """
    try:
        U, svals, Vt = np.linalg.svd(Jv, full_matrices=False)
    except Exception:
        # Fallback: calcular mediante eig de J J^T
        M = Jv @ Jv.T
        M = 0.5 * (M + M.T)
        eigvals, eigvecs = np.linalg.eigh(M)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        axes = np.sqrt(np.clip(eigvals, 0, None))
        return axes, eigvecs, M, np.array([]), np.inf

    M = Jv @ Jv.T
    cond = np.inf
    if svals.size > 0 and svals[-1] > 0:
        cond = svals[0] / svals[-1]
    # Asegurar 3 ejes (si Jv es 3xN, svals tiene longitud 3 como mucho)
    # Si hay menos de 3 valores (raro), rellenar con ceros
    if svals.size < 3:
        svals = np.pad(svals, (0, 3 - svals.size), 'constant')
        # Y completar U con columnas ortonormales arbitrarias:
        if U.shape[1] < 3:
            # completar con vectores estándar y ortonormalizar
            extra = np.eye(3)[:, :3 - U.shape[1]]
            U = np.concatenate((U, extra), axis=1)
            # re-ortonormalizar con QR
            Q, _ = np.linalg.qr(U)
            U = Q

    return svals, U, M, svals, cond

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

def q0_from_config(config):
    s, e, w = config
    q0 = np.zeros(6)
    q0[0] = -np.pi/4 if s == 0 else np.pi/4
    q0[2] = -np.pi/4 if e == 0 else np.pi/4
    q0[4] = 0 if w == 0 else np.pi/2
    return q0

# ------------------------- PyQtGraph GUI -------------------------
class ManipulabilityAppGL(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('IRB140 - Elipsoide de manipulabilidad (PyQtGraph)')
        self.joints = np.zeros(DOF)
        self.playing = False
        self.traj = []
        self.traj_idx = 0
        self.use_full_jacobian_monitor = USE_FULL_JACOBIAN_IF_RTB and (robot is not None)
        self.planned_pts = None
        self.planned_markers = None
        self.active_config = None

        # Layout
        layout = QtWidgets.QHBoxLayout(self)
        ctrl = QtWidgets.QVBoxLayout()
        layout.addLayout(ctrl, 1)

        # Labels de origen y destino (inicialmente vacíos)
        self.origin_label = QtWidgets.QLabel("Origen: -")
        self.origin_label.setStyleSheet("color: red; font-weight: bold;")
        ctrl.addWidget(self.origin_label)

        self.dest_label = QtWidgets.QLabel("Destino: -")
        self.dest_label.setStyleSheet("color: orange; font-weight: bold;")
        ctrl.addWidget(self.dest_label)

        # Sliders + labels
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

        # Trajectory buttons
        self.traj_btn = QtWidgets.QPushButton('Generar trayectoria rectilínea (origen->destino)')
        self.traj_btn.clicked.connect(self.generate_trajectory_with_origin_dest)
        ctrl.addWidget(self.traj_btn)

        # Play/Stop
        hplay = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton('Play traj')
        self.play_btn.clicked.connect(self.play_pause)
        self.stop_btn = QtWidgets.QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop)
        hplay.addWidget(self.play_btn)
        hplay.addWidget(self.stop_btn)
        ctrl.addLayout(hplay)

        # Speed slider
        self.speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.change_speed)
        ctrl.addWidget(QtWidgets.QLabel("Velocidad animación"))
        ctrl.addWidget(self.speed_slider)
        self.speed_factor = 1.0

        # Info label
        self.info_label = QtWidgets.QLabel('Condición: -')
        ctrl.addWidget(self.info_label)

        # Full jacobian monitor checkbox
        if robot is not None:
            self.full_jac_chk = QtWidgets.QCheckBox('Monitor jacobiana completa 6xN (RTB)')
            self.full_jac_chk.setChecked(self.use_full_jacobian_monitor)
            self.full_jac_chk.stateChanged.connect(self.toggle_full_jac)
            ctrl.addWidget(self.full_jac_chk)

        # View buttons
        view_layout = QtWidgets.QHBoxLayout()
        btn_top = QtWidgets.QPushButton("Vista superior")
        btn_front = QtWidgets.QPushButton("Vista frontal")
        btn_side = QtWidgets.QPushButton("Vista lateral")
        btn_iso = QtWidgets.QPushButton("Vista isométrica")
        view_layout.addWidget(btn_top)
        view_layout.addWidget(btn_front)
        view_layout.addWidget(btn_side)
        view_layout.addWidget(btn_iso)
        ctrl.addLayout(view_layout)
        btn_top.clicked.connect(lambda: self.set_view('top'))
        btn_front.clicked.connect(lambda: self.set_view('front'))
        btn_side.clicked.connect(lambda: self.set_view('side'))
        btn_iso.clicked.connect(lambda: self.set_view('iso'))

        # Axis limit checkbox (implemented as camera-fix toggle)
        self.axis_limit_chk = QtWidgets.QCheckBox("Usar cámara fija (limites)")
        self.axis_limit_chk.setChecked(True)
        ctrl.addWidget(self.axis_limit_chk)

        # Axis info: mostrar referencias de ejes y escala
        self.axis_info_label = QtWidgets.QLabel("Ejes: X+ red, Y+ green, Z+ blue | Tick = 0.1 m")
        self.axis_info_label.setStyleSheet("color: white;")
        ctrl.addWidget(self.axis_info_label)

        ctrl.addStretch(1)

        # GL View
        self.view = gl.GLViewWidget()
        layout.addWidget(self.view, 3)  # <-- Agregar la vista 3D al layout

        self.view.opts['distance'] = 1.5
        self.view.setBackgroundColor('k')   # negro


        # Posición inicial de cámara (isométrica)
        self.view.setCameraPosition(distance=1.5, elevation=30, azimuth=45)

        # Grid (piso)
        g = gl.GLGridItem()
        g.setSize(2.0, 2.0)
        g.setSpacing(0.1, 0.1)
        g.translate(0, 0, 0)   # piso en z=0
        self.view.addItem(g)


        # --- Cargar STL de los links ---
        self.link_meshes = []
        stl_files = [
            "visual_simplified/base_link.stl",
            "visual_simplified/link_1.stl",
            "visual_simplified/link_2.stl",
            "visual_simplified/link_3.stl",
            "visual_simplified/link_4.stl",
            "visual_simplified/link_5.stl",
            "visual_simplified/link_6.stl",
        ]
        for f in stl_files:
            try:
                m = mesh.Mesh.from_file(f)
                verts = np.vstack([m.v0, m.v1, m.v2]) / 50.0   # mm → m
                faces = np.arange(len(verts)).reshape(-1, 3)
                md = gl.MeshData(vertexes=verts, faces=faces)
                #md.computeNormals()
                item = gl.GLMeshItem(
                    meshdata=md,
                    smooth=True,
                    color=(0.7, 0.7, 0.7, 1.0),
                    shader="shaded",
                    glOptions="opaque"
                )
                item.setTransform(QtGui.QMatrix4x4())  # identidad
                self.view.addItem(item)
                self.link_meshes.append(item)
            except Exception as e:
                print(f"Error cargando STL {f}: {e}")
                self.link_meshes.append(None)


        # Robot line
        self.robot_line = gl.GLLinePlotItem(width=4, antialias=True, color=(0, 0.5, 1, 1), mode='line_strip')
        self.view.addItem(self.robot_line)

        # Joints scatter (pequeñas esferas rojas, tamaño en pixels)
        self.joint_scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=6, pxMode=True, color=(1, 0, 0, 1))  # rojo
        self.view.addItem(self.joint_scatter)

        # Terna de la TOOL (ejes en azul claro)
        self.tcp_axis_x = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.4, 0.75, 1, 1), width=3)
        self.tcp_axis_y = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.4, 0.75, 1, 1), width=3)
        self.tcp_axis_z = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.4, 0.75, 1, 1), width=3)
        self.view.addItem(self.tcp_axis_x)
        self.view.addItem(self.tcp_axis_y)
        self.view.addItem(self.tcp_axis_z)

        # Elipsoide: mesh (se recrea), y vectores principales (cian)
        self.ellipsoid_mesh_item = None
        self.ellipsoid_vecs = [
            gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0, 1, 1, 1), width=2),
            gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0, 1, 1, 1), width=2),
            gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0, 1, 1, 1), width=2),
        ]
        for v in self.ellipsoid_vecs:
            self.view.addItem(v)

        # Ejes globales (X red, Y green, Z blue) y ticks simples
        axis_len = 1.0
        self.axis_x = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [axis_len, 0, 0]], dtype=np.float32), color=(1, 0, 0, 1), width=2)
        self.axis_y = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, axis_len, 0]], dtype=np.float32), color=(0, 1, 0, 1), width=2)
        self.axis_z = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, axis_len]], dtype=np.float32), color=(0, 0, 1, 1), width=2)
        self.view.addItem(self.axis_x)
        self.view.addItem(self.axis_y)
        self.view.addItem(self.axis_z)

        # Tick marks (lines) for axes (static ticks at 0.1,0.2,...)
        ticks = []
        tickstep = 0.1
        nticks = int(axis_len / tickstep)
        for i in range(1, nticks + 1):
            x = i * tickstep
            ticks.append(gl.GLLinePlotItem(pos=np.array([[x, -0.01, 0], [x, 0.01, 0]], dtype=np.float32), color=(1, 0, 0, 1), width=1))
            y = i * tickstep
            ticks.append(gl.GLLinePlotItem(pos=np.array([[-0.01, y, 0], [0.01, y, 0]], dtype=np.float32), color=(0, 1, 0, 1), width=1))
            z = i * tickstep
            ticks.append(gl.GLLinePlotItem(pos=np.array([[-0.01, 0, z], [0.01, 0, z]], dtype=np.float32), color=(0, 0, 1, 1), width=1))
        for t in ticks:
            self.view.addItem(t)

        # Recta de trayectoria rectilínea
        self.planned_line = gl.GLLinePlotItem(pos=np.zeros((2, 3), dtype=np.float32), width=2, antialias=True, color=(1, 1, 0, 1))  # amarillo
        self.view.addItem(self.planned_line)

        # Puntos de la trayectoria: puntos intermedios verdes pequeños (pxMode True)
        self.planned_scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=4, pxMode=True, color=(0, 1, 0, 1))  # verdes
        self.view.addItem(self.planned_scatter)

        # Puntos extremos (origen/destino) como esferas rojas pequeñas (pxMode True)
        self.origin_scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=9, pxMode=True, color=(1, 0, 0, 1))
        self.view.addItem(self.origin_scatter)
        self.dest_scatter = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=9, pxMode=True, color=(1, 0, 0, 1))
        self.view.addItem(self.dest_scatter)

        # Init scene
        self.update_scene(initial=True)

        # Timer for animation
        self.timer = QtCore.QTimer()
        base_interval = 20  # ms
        self.timer.setInterval(int(base_interval / self.speed_factor))
        self.timer.timeout.connect(self._play_step)
        # not started until user presses play



    # ---------------- GUI helpers ----------------
    def toggle_full_jac(self, state):
        self.use_full_jacobian_monitor = (state == QtCore.Qt.Checked)

    def set_view(self, mode):
        if mode == 'top':
            self.view.setCameraPosition(elevation=90, azimuth=-90, distance=self.view.opts['distance'])
        elif mode == 'front':
            self.view.setCameraPosition(elevation=0, azimuth=0, distance=self.view.opts['distance'])
        elif mode == 'side':
            self.view.setCameraPosition(elevation=0, azimuth=90, distance=self.view.opts['distance'])
        elif mode == 'iso':
            self.view.setCameraPosition(elevation=30, azimuth=45, distance=self.view.opts['distance'])


    def change_speed(self, val):
        # val in [0,100]
        min_interval = 150   # más lento
        max_interval = 10    # más rápido, pero no tan brusco
        interval = int(min_interval - (val/100.0) * (min_interval - max_interval))
        self.timer.setInterval(interval)


    def make_slider_callback(self, idx, label):
        def cb(val):
            angle = np.deg2rad(val)
            label.setText(f'Joint {idx+1}: {angle:.3f} rad')
            self.joints[idx] = angle
            self.joints = clip_joints(self.joints)

            # --- borrar trayectoria planificada si existe ---
            if self.planned_pts is not None:
                self.planned_pts = None
                self.planned_markers = None
                # vaciar la recta y los puntos intermedios
                self.planned_line.setData(pos=np.zeros((2,3), dtype=np.float32))
                self.planned_scatter.setData(pos=np.zeros((1,3), dtype=np.float32))
                # ---- y vaciar también los puntos extremos (origen/destino) ----
                self.origin_scatter.setData(pos=np.zeros((1,3), dtype=np.float32))
                self.dest_scatter.setData(pos=np.zeros((1,3), dtype=np.float32))

            self.update_scene()
        return cb


    def update_scene(self, initial=False):
        # Cinemática
        try:
            Ttcp, Ts = forward_kinematics(self.joints)
        except Exception as e:
            print("FK Error:", e)
            return

        origins = np.array([T[:3, 3] for T in Ts], dtype=np.float32)

        # Robot geometry
        if origins.size > 0:
            self.robot_line.setData(pos=origins)
            self.joint_scatter.setData(pos=origins, size=6, pxMode=True)
        else:
            self.robot_line.setData(pos=np.zeros((2, 3), dtype=np.float32))
            self.joint_scatter.setData(pos=np.zeros((1, 3), dtype=np.float32), size=6, pxMode=True)

        # --- Actualizar posición de los meshes STL ---
        try:
            for i, item in enumerate(self.link_meshes):
                if item is None:
                    continue
                if i < len(Ts):
                    T = Ts[i]
                    M = np.eye(4, dtype=np.float32)
                    M[:3, :3] = T[:3, :3]
                    M[:3, 3] = T[:3, 3]
                    mat = QtGui.QMatrix4x4(*M.T.flatten())
                    item.setTransform(mat)
        except Exception as e:
            print("Error update STL:", e)


        # Compute TCP pose
        p = Ttcp[:3, 3]
        R = Ttcp[:3, :3]

        # Terna TOOL: actualizar líneas (usar R columnas)
        try:
            axis_len = 0.12
            x_end = (p + axis_len * R[:, 0]).astype(np.float32)
            y_end = (p + axis_len * R[:, 1]).astype(np.float32)
            z_end = (p + axis_len * R[:, 2]).astype(np.float32)
            self.tcp_axis_x.setData(pos=np.vstack((p.astype(np.float32), x_end)))
            self.tcp_axis_y.setData(pos=np.vstack((p.astype(np.float32), y_end)))
            self.tcp_axis_z.setData(pos=np.vstack((p.astype(np.float32), z_end)))
        except Exception:
            pass

        # Planned trajectory (persist until sliders change)
        if self.planned_pts is not None and len(self.planned_pts) > 0:
            pts = np.array(self.planned_pts, dtype=np.float32)
            self.planned_line.setData(pos=pts, mode='line_strip')
            if pts.shape[0] > 2:
                self.planned_scatter.setData(pos=pts[1:-1], size=4, pxMode=True)
            else:
                self.planned_scatter.setData(pos=np.zeros((1, 3), dtype=np.float32))
            self.origin_scatter.setData(pos=np.array([pts[0]], dtype=np.float32), size=9, pxMode=True)
            self.dest_scatter.setData(pos=np.array([pts[-1]], dtype=np.float32), size=9, pxMode=True)
        else:
            self.planned_line.setData(pos=np.zeros((2, 3), dtype=np.float32))
            self.planned_scatter.setData(pos=np.zeros((1, 3), dtype=np.float32))
            self.origin_scatter.setData(pos=np.zeros((1, 3), dtype=np.float32))
            self.dest_scatter.setData(pos=np.zeros((1, 3), dtype=np.float32))

        # ---------------- Elipsoide ----------------
        Jv = numeric_jacobian(self.joints)
        axes_len, axes_dir, M, svals, cond = manipulability_ellipsoid_from_J(Jv)

        p = Ttcp[:3, 3].astype(np.float32)   # posición TCP

        # chequeo: si la elipsoide está degenerada, no la dibujamos
        if np.min(axes_len) < 1e-8 or np.any(np.isnan(axes_len)):
            if self.ellipsoid_mesh_item is not None:
                try:
                    self.view.removeItem(self.ellipsoid_mesh_item)
                except Exception:
                    pass
                self.ellipsoid_mesh_item = None
            # también borrar los vectores principales
            for v in self.ellipsoid_vecs:
                v.setData(pos=np.zeros((2, 3), dtype=np.float32))
        else:
            # generar esfera paramétrica
            nu, nv = 28, 14
            u = np.linspace(0, 2*np.pi, nu)
            v = np.linspace(0, np.pi, nv)
            X = np.outer(np.cos(u), np.sin(v))
            Y = np.outer(np.sin(u), np.sin(v))
            Z = np.outer(np.ones_like(u), np.cos(v))

            verts = []
            for i in range(nu):
                for j in range(nv):
                    vec = np.array([X[i, j], Y[i, j], Z[i, j]], dtype=np.float32)
                    point = (axes_len[0] * vec[0] * axes_dir[:, 0] +
                            axes_len[1] * vec[1] * axes_dir[:, 1] +
                            axes_len[2] * vec[2] * axes_dir[:, 2])
                    verts.append((p + point).astype(np.float32))
            verts = np.nan_to_num(np.array(verts, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

            # faces (triangulación)
            faces = []
            for i in range(nu - 1):
                for j in range(nv - 1):
                    a = i * nv + j
                    b = (i + 1) * nv + j
                    c = i * nv + (j + 1)
                    d = (i + 1) * nv + (j + 1)
                    faces.append([a, b, c])
                    faces.append([b, d, c])
            faces = np.array(faces, dtype=np.int32)

            try:
                meshdata = gl.MeshData(vertexes=verts, faces=faces)

                if self.ellipsoid_mesh_item is not None:
                    try:
                        self.view.removeItem(self.ellipsoid_mesh_item)
                    except Exception:
                        pass

                self.ellipsoid_mesh_item = gl.GLMeshItem(
                    meshdata=meshdata,
                    smooth=True,
                    drawFaces=True,
                    drawEdges=False,
                    shader='shaded',
                    glOptions='translucent'
                )
                # verde claro translúcido
                self.ellipsoid_mesh_item.setColor((0.35, 0.95, 0.1, 0.5))
                self.view.addItem(self.ellipsoid_mesh_item)
            except Exception as e:
                # si algo falla, eliminamos el mesh
                if self.ellipsoid_mesh_item is not None:
                    try:
                        self.view.removeItem(self.ellipsoid_mesh_item)
                    except Exception:
                        pass
                self.ellipsoid_mesh_item = None

        # Vectores principales de la elipsoide (cian)
        try:
            for i in range(3):
                start = p
                end = (p + axes_len[i] * axes_dir[:, i]).astype(np.float32)
                self.ellipsoid_vecs[i].setData(pos=np.vstack((start, end)))
        except Exception:
            for v in self.ellipsoid_vecs:
                v.setData(pos=np.zeros((2, 3), dtype=np.float32))


        # Info label: obtener s_min y lambda usado si posible
        try:
            s_min = svals[-1] if svals.size > 0 else 0.0
        except Exception:
            s_min = 0.0

        try:
            _, lam, svals_tmp = damped_pinv_adaptive(Jv)
        except Exception:
            lam = 0.0

        info_text = f's_min_pos={s_min:.3e} | λ={lam:.3e} | cond={cond:.3e}'
        self.info_label.setText(info_text)

        # If initial, set an isometric camera
        if initial:
            self.set_view('iso')


    # ---------------- Trajectory generation / playback ----------------
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
            if config_text is not None and len(config_text.strip())>0:
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
        self.planned_pts = planned_pts
        self.planned_markers = [('Origen', p0), ('Destino', pd)]

        self.origin_label.setText(f"Origen: {np.round(p0, 3)}")
        self.dest_label.setText(f"Destino: {np.round(pd, 3)}")


        # --- Labels de origen y destino ---
        if hasattr(self, "origin_label"):
            self.origin_label.setText(f"Origen: {p0}")
        else:
            self.origin_label = QtWidgets.QLabel(f"Origen: {p0}")
            self.layout().itemAt(0).layout().addWidget(self.origin_label)  # ctrl layout

        if hasattr(self, "dest_label"):
            self.dest_label.setText(f"Destino: {pd}")
        else:
            self.dest_label = QtWidgets.QLabel(f"Destino: {pd}")
            self.layout().itemAt(0).layout().addWidget(self.dest_label)


        # mostrar mensaje y actualizar vista
        QtWidgets.QMessageBox.information(self, 'Trayectoria', f'Trayectoria generada: {len(full_traj)} pasos.')
        self.update_scene()

    def _play_step(self):
        # Si no hay trayectoria o ya terminó
        if not self.traj or self.traj_idx >= len(self.traj):
            self.timer.stop()
            self.playing = False
            self.play_btn.setText('Play traj')
            for s in self.sliders:
                s.setEnabled(True)
            # Al terminar la ejecución, liberamos la active_config
            self.active_config = None
            return

        # Tomamos el siguiente paso precomputado
        q_pre, lam, s_min_pos, s_min_full, cond_full = self.traj[self.traj_idx]
        self.traj_idx += 1

        # Si hay una configuración activa pedida, intentamos forzar la IK desde esa config
        q_to_use = q_pre.copy()
        if self.active_config is not None:
            try:
                # obtener el target TCP asociado al q_pre (posición a alcanzar en este paso)
                T_target, _ = forward_kinematics(q_pre)
                p_target = T_target[:3, 3]

                # construir semilla a partir de la configuración solicitada
                q_seed = q0_from_config(self.active_config)

                # intentar resolver IK desde esa semilla (use_full_jacobian arg opcional según tu código)
                q_sol, ok_conv, its = solve_ik_to_point(q_seed, p_target, max_iters=800, tol=1e-5, use_full_jacobian=self.use_full_jacobian_monitor)

                if ok_conv:
                    q_to_use = q_sol  # si convergió, preferimos la solución que cumple la config solicitada
                # si no convergió, cleverd fallback: usar el q_pre que ya teníamos
            except Exception as e:
                # en caso de error, mantener q_pre
                # print("IK (active_config) error:", e)
                q_to_use = q_pre.copy()

        # Aplicar y mostrar la configuración escogida
        self.joints = clip_joints(q_to_use)

        # Deshabilitar sliders mientras se reproduce la trayectoria
        for s in self.sliders:
            s.setEnabled(False)

        # Actualizar sliders y labels (sin emitir señales)
        for i, s in enumerate(self.sliders):
            s.blockSignals(True)
            s.setValue(int(np.rad2deg(self.joints[i])))
            self.labels[i].setText(f'Joint {i+1}: {self.joints[i]:.3f} rad')
            s.blockSignals(False)

        # Actualizar escena y la info del paso
        self.update_scene()

        info_text = f's_min_step={s_min_pos:.3e} | λ={lam:.3e}'
        if s_min_full is not None:
            info_text += f' | s_min_full={s_min_full:.3e}'
        if cond_full is not None:
            info_text += f' | cond_full={cond_full:.3e}'
        self.info_label.setText(info_text)


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
    w = ManipulabilityAppGL()
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
