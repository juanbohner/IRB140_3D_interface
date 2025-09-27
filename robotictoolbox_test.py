import numpy as np
if not hasattr(np, "disp"):
        np.disp = lambda x: print(x)
import roboticstoolbox as rtb
from spatialmath import SE3

# Cargar el modelo del robot IRB140
robot = rtb.models.DH.IRB140()

print(robot)

# Ejemplo: cinemática directa de una configuración articular
q = [0, -np.pi/4, np.pi/4, 0, np.pi/6, 0]  # en radianes
T = robot.fkine(q)
print(T)
print("T.A:\n", T.A)
print("\nTransformación homogénea en el efector final:")
print(T)

# Ejemplo: cinemática inversa hacia un punto deseado
Td = SE3(0.4, 0.2, 0.5) * SE3.Rx(np.pi)  # Pose deseada
q_sol = robot.ikine_LM(Td)  # inversa con Levenberg-Marquardt

print("\nSolución de la cinemática inversa:")
print(q_sol)

# Visualizar el robot en esa configuración
robot.plot(q, block=True)
