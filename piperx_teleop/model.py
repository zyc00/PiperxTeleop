"""Rigid-body gravity model for the AgileX PiPER-X arm, from the vendor URDF.

The bundled URDF is piper_x + gripper (agilexrobotics/agx_arm_urdf@main): this
arm's firmware FK matches it to 0.01 mm. Do NOT swap in the plain-piper file
- its wrist COMs are a different robot (J4 haunting, J5 runaways).

Joint convention matches the SDK exactly: q[i] is the URDF joint variable in
radians, and the SDK reports/accepts the same quantity in units of 0.001 deg
(see piper_ctrl_single_node.py in piper_ros, which converts with no offset or
sign flip).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

URDF_PATH = os.path.join(os.path.dirname(__file__), "piper_x_description.urdf")
GRAVITY = np.array([0.0, 0.0, -9.80665])
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def rpy_to_R(r, p, y):
    """Fixed-axis roll-pitch-yaw, i.e. R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def _T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


class Joint:
    def __init__(self, el):
        self.name = el.get("name")
        self.type = el.get("type")
        self.parent = el.find("parent").get("link")
        self.child = el.find("child").get("link")
        o = el.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0")).split()] if o is not None else [0, 0, 0]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0")).split()] if o is not None else [0, 0, 0]
        self.T_origin = _T(rpy_to_R(*rpy), np.array(xyz))
        a = el.find("axis")
        self.axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.array([0.0, 0.0, 1.0])

    def T_motion(self, q):
        if self.type == "revolute" or self.type == "continuous":
            return _T(_axis_angle_R(self.axis, q), np.zeros(3))
        if self.type == "prismatic":
            return _T(np.eye(3), self.axis * q)
        return np.eye(4)


def _axis_angle_R(axis, q):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)


class PiperModel:
    """Gravity torques for the 6 arm joints, with an optional tool payload."""

    def __init__(self, urdf_path=URDF_PATH, payload_mass=0.0, payload_com=(0.0, 0.0, 0.0),
                 payload_link="gripper_base"):
        root = ET.parse(urdf_path).getroot()
        self.joints = {}
        self.children = {}
        for el in root.findall("joint"):
            j = Joint(el)
            self.joints[j.name] = j
            self.children.setdefault(j.parent, []).append(j.name)

        self.links = {}   # link -> (mass, com in link frame)
        for el in root.findall("link"):
            ine = el.find("inertial")
            if ine is None:
                self.links[el.get("name")] = (0.0, np.zeros(3))
                continue
            m = float(ine.find("mass").get("value"))
            o = ine.find("origin")
            com = np.array([float(v) for v in o.get("xyz").split()]) if o is not None else np.zeros(3)
            self.links[el.get("name")] = (m, com)

        # Fold a tool payload into the link it is bolted to.
        if payload_mass > 0:
            m0, c0 = self.links[payload_link]
            m_new = m0 + payload_mass
            c_new = (m0 * c0 + payload_mass * np.array(payload_com)) / m_new
            self.links[payload_link] = (m_new, c_new)

        children_links = {j.child for j in self.joints.values()}
        self.root_link = next(l for l in self.links if l not in children_links)
        self.total_mass = sum(m for m, _ in self.links.values())

    def _walk(self, q, gripper=0.0):
        """Return (T_link[world], T_jointframe[world]) for every link/joint."""
        T_link = {self.root_link: np.eye(4)}
        T_jf = {}
        stack = [self.root_link]
        while stack:
            parent = stack.pop()
            for jname in self.children.get(parent, []):
                j = self.joints[jname]
                Tj = T_link[parent] @ j.T_origin      # joint frame, before motion
                T_jf[jname] = Tj
                if jname in ARM_JOINTS:
                    qi = q[ARM_JOINTS.index(jname)]
                elif j.type == "prismatic":
                    qi = gripper if j.axis[2] > 0 else -gripper
                else:
                    qi = 0.0
                T_link[j.child] = Tj @ j.T_motion(qi)
                stack.append(j.child)
        return T_link, T_jf

    def _subtree_links(self, jname):
        out, stack = [], [self.joints[jname].child]
        while stack:
            l = stack.pop()
            out.append(l)
            for c in self.children.get(l, []):
                stack.append(self.joints[c].child)
        return out

    def potential_energy(self, q, gripper=0.0):
        T_link, _ = self._walk(q, gripper)
        U = 0.0
        for lname, (m, com) in self.links.items():
            if m == 0:
                continue
            p = T_link[lname] @ np.append(com, 1.0)
            U -= m * GRAVITY @ p[:3]
        return U

    def gravity_torque(self, q, gripper=0.0):
        """Static holding torque (N.m) the motors must apply, per arm joint."""
        T_link, T_jf = self._walk(q, gripper)
        p_com = {}
        for lname, (m, com) in self.links.items():
            p_com[lname] = (T_link[lname] @ np.append(com, 1.0))[:3]

        tau = np.zeros(6)
        for i, jname in enumerate(ARM_JOINTS):
            Tj = T_jf[jname]
            o_i = Tj[:3, 3]
            z_i = Tj[:3, :3] @ self.joints[jname].axis
            s = 0.0
            for lname in self._subtree_links(jname):
                m, _ = self.links[lname]
                if m == 0:
                    continue
                s += m * GRAVITY @ _cross(z_i, p_com[lname] - o_i)
            tau[i] = -s
        return tau

    def gravity_torque_fd(self, q, gripper=0.0, eps=1e-6):
        """Finite-difference reference implementation, for validation only."""
        tau = np.zeros(6)
        for i in range(6):
            qp, qm = np.array(q, float), np.array(q, float)
            qp[i] += eps
            qm[i] -= eps
            tau[i] = (self.potential_energy(qp, gripper) - self.potential_energy(qm, gripper)) / (2 * eps)
        return tau

    def fk(self, q, link="gripper_base", gripper=0.0):
        T_link, _ = self._walk(q, gripper)
        return T_link[link]


# --- gravity parameter identification -------------------------------------
#
# Gravity torque is linear in each link's barycentric parameters
# (m, m*cx, m*cy, m*cz):
#
#     U = sum_k [ m_k * (-g . p_k(q)) + (m_k c_k) . (-R_k(q)^T g) ]
#     tau_i = dU/dq_i
#
# so tau = Y(q) @ beta.  Building Y lets us re-fit beta against torques measured
# on the real arm instead of trusting the URDF, which is badly wrong at the
# wrist (J4 shows ~2.4 N.m where the URDF predicts ~0).

    def _link_basis(self, q, links, gripper=0.0):
        """Potential-energy basis functions, one row per (link, parameter)."""
        T_link, _ = self._walk(q, gripper)
        out = []
        for lname in links:
            T = T_link[lname]
            out.append(-GRAVITY @ T[:3, 3])          # coefficient of m
            v = -(T[:3, :3].T @ GRAVITY)             # coefficients of m*c
            out.extend(v.tolist())
        return np.array(out)

    def gravity_regressor(self, q, links=None, gripper=0.0, eps=1e-5):
        """Y(q) with shape (6, 4*len(links)) such that tau = Y @ beta."""
        links = list(links if links is not None else self.identifiable_links())
        cols = 4 * len(links)
        Y = np.zeros((6, cols))
        for i in range(6):
            qp = np.array(q, float); qm = np.array(q, float)
            qp[i] += eps; qm[i] -= eps
            Y[i] = (self._link_basis(qp, links, gripper) -
                    self._link_basis(qm, links, gripper)) / (2 * eps)
        return Y

    def identifiable_links(self):
        """Links that can carry gravity load, ordered outward from the base.

        Names follow the piper_x + gripper URDF (link7/link8 were the old
        piper file's finger names; they no longer exist)."""
        return ["link1", "link2", "link3", "link4", "link5", "link6",
                "flange_link", "gripper_base", "gripper_link1", "gripper_link2"]

    def beta_urdf(self, links=None):
        """The URDF's own barycentric parameters, as the regressor orders them."""
        links = list(links if links is not None else self.identifiable_links())
        b = []
        for lname in links:
            m, c = self.links[lname]
            b.extend([m, m * c[0], m * c[1], m * c[2]])
        return np.array(b)

    def subtree_link_index(self, joint_idx, links=None):
        """Column indices of links distal to a joint - the only ones it feels."""
        links = list(links if links is not None else self.identifiable_links())
        distal = set(self._subtree_links(ARM_JOINTS[joint_idx]))
        idx = []
        for k, lname in enumerate(links):
            if lname in distal:
                idx.extend(range(4 * k, 4 * k + 4))
        return np.array(idx, int)
