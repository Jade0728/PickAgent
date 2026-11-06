# 역할: 장면 생성·IK·팔과 집게 제어·도달 검사·집기 작업을 담당합니다.

import math
import threading
import time

import mujoco
import mujoco_menagerie as menagerie
import numpy as np

from . import config as cfg


class OperationCancelled(RuntimeError):
    """사용자 중단 또는 뷰어 종료를 일반 실패와 구분합니다."""


def find_id(model, object_type, name):
    """이름으로 모델 요소를 조회하고 잘못된 이름은 거부합니다."""
    object_id = mujoco.mj_name2id(model, object_type, name)

    if object_id < 0:
        raise ValueError(f"모델 요소를 찾을 수 없습니다: {name}")

    return object_id


def skew(vector):
    """벡터의 외적을 행렬 곱으로 표현합니다."""
    x, y, z = vector

    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ])


def build_scene():
    """SO-101 모델에 큐브·목적지·고정 카메라를 추가합니다."""
    spec = menagerie.get("robotstudio_so101").spec("scene")

    cube = spec.worldbody.add_body(
        name="blue_cube",
        pos=cfg.PICK.tolist(),
    )
    cube.add_freejoint()
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cfg.CUBE_HALF] * 3,
        mass=0.03,
        rgba=[0.1, 0.3, 0.95, 1],
        friction=[1.5, 0.01, 0.001],
        condim=6,
    )

    # site는 시각 표시이며 물체와 충돌하지 않습니다.
    spec.worldbody.add_site(
        name="place_marker",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        pos=[cfg.PLACE[0], cfg.PLACE[1], 0.001],
        size=[0.035, 0.001, 0],
        rgba=[0.1, 0.9, 0.2, 0.5],
    )

    spec.worldbody.add_camera(
        name=cfg.CAMERA_NAME,
        pos=[0.20, 0.04, 0.50],
        quat=[1, 0, 0, 0],
        fovy=40,
    )

    model = spec.compile()
    model.opt.timestep = 0.002

    # 바닥 색과 반사상이 이미지 인식을 방해하지 않게 조정합니다.
    floor_id = find_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
    )
    model.geom_matid[floor_id] = -1
    model.geom_rgba[floor_id] = [0.70, 0.70, 0.70, 1]
    model.mat_reflectance[:] = 0

    return model


class Robot:
    """시뮬레이션 상태와 로봇 제어에 필요한 정보를 관리합니다."""

    def __init__(self, cancel_event=None):
        """장면과 관절 인덱스를 준비하고 관찰 자세로 초기화합니다."""
        self.model = build_scene()
        self.data = mujoco.MjData(self.model)
        self.ik_data = mujoco.MjData(self.model)

        self.cancel_event = (
            cancel_event
            if cancel_event is not None
            else threading.Event()
        )

        self.arm_ids = np.array([
            find_id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            for name in cfg.ARM_NAMES
        ])
        self.grip_id = find_id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
        )
        self.site_id = find_id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe"
        )
        self.cube_id = find_id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "blue_cube"
        )

        joints = self.model.actuator_trnid[self.arm_ids, 0]
        self.qpos_ids = self.model.jnt_qposadr[joints]
        self.dof_ids = self.model.jnt_dofadr[joints]

        self.lower = np.maximum(
            self.model.jnt_range[joints, 0],
            self.model.actuator_ctrlrange[self.arm_ids, 0],
        )
        self.upper = np.minimum(
            self.model.jnt_range[joints, 1],
            self.model.actuator_ctrlrange[self.arm_ids, 1],
        )

        # 집게 본체의 접근 방향을 site 좌표계로 변환합니다.
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(
            rotation, self.model.site_quat[self.site_id]
        )
        self.approach_local = (
            rotation.reshape(3, 3).T
            @ np.array([0.0, 0.0, -1.0])
        )

        self.rng = np.random.default_rng(7)
        self.faulted = False
        self.stage = "idle"
        self.last_ik_metrics = None
        self.last_reach_metrics = None

        self._initialize_pose()

    def _initialize_pose(self):
        """초기화 시에만 관절 위치를 직접 설정합니다."""
        initial = np.radians([60, -60, 60, 0, 0])

        self.data.qpos[self.qpos_ids] = initial
        self.data.ctrl[self.arm_ids] = initial

        grip_joint = int(
            self.model.actuator_trnid[self.grip_id, 0]
        )
        address = self.model.jnt_qposadr[grip_joint]

        self.data.qpos[address] = cfg.GRIP_OPEN
        self.data.ctrl[self.grip_id] = cfg.GRIP_OPEN

        mujoco.mj_forward(self.model, self.data)

    def reset(self, viewer):
        """사용자가 요청한 경우 시뮬레이션을 초기 상태로 되돌립니다."""
        self.check_operation(viewer)
        self.faulted = True

        mujoco.mj_resetData(self.model, self.data)
        self._initialize_pose()

        self.last_ik_metrics = None
        self.last_reach_metrics = None
        self.stage = "reset"

        self.hold(viewer, 1.5)

        self.faulted = False
        self.stage = "idle"

    def check_operation(self, viewer=None):
        """중단 이벤트와 뷰어 종료를 검사합니다."""
        if self.cancel_event.is_set():
            raise OperationCancelled("사용자가 작업을 중단했습니다.")

        if viewer is not None and not viewer.is_running():
            raise OperationCancelled("뷰어가 닫혔습니다.")

    def hold_current_arm(self):
        """팔 목표를 현재 자세로 바꾸되 집게 목표는 유지합니다."""
        self.data.ctrl[self.arm_ids] = np.clip(
            self.data.qpos[self.qpos_ids],
            self.lower,
            self.upper,
        )

    def step(self, viewer):
        """중단 여부를 확인하며 물리와 화면을 한 단계 갱신합니다."""
        self.check_operation(viewer)
        started = time.perf_counter()

        mujoco.mj_step(self.model, self.data)
        viewer.sync()

        self.check_operation(viewer)

        remaining = (
            self.model.opt.timestep
            - (time.perf_counter() - started)
        )
        if remaining > 0:
            time.sleep(remaining)

    def hold(self, viewer, seconds):
        """현재 목표를 지정한 시뮬레이션 시간 동안 유지합니다."""
        self.check_operation(viewer)
        deadline = self.data.time + seconds

        while self.data.time < deadline:
            self.step(viewer)

        self.check_operation(viewer)

    def move_controls(self, viewer, target, seconds):
        """현재 제어값에서 목표 제어값까지 부드럽게 보간합니다."""
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("이동 시간은 유한한 양수여야 합니다.")

        self.check_operation(viewer)

        start = self.data.ctrl.copy()
        started = self.data.time

        while True:
            self.check_operation(viewer)

            progress = min(
                (self.data.time - started) / seconds, 1.0
            )
            smooth = progress**2 * (3 - 2 * progress)
            self.data.ctrl[:] = start + smooth * (target - start)

            self.step(viewer)

            if progress >= 1.0:
                return

    def set_gripper(self, viewer, angle):
        """팔 목표를 유지하며 집게만 움직입니다."""
        lower, upper = self.model.actuator_ctrlrange[self.grip_id]

        if not lower <= angle <= upper:
            raise ValueError("집게 각도가 제어 범위를 벗어났습니다.")

        target = self.data.ctrl.copy()
        target[self.grip_id] = angle

        self.move_controls(viewer, target, 1.5)
        self.hold(viewer, 0.5)

    def solve_ik(self, target, seed):
        """별도 데이터에서 목표 위치와 아래쪽 접근 방향을 계산합니다."""
        target = np.asarray(target, dtype=float)
        seed = np.asarray(seed, dtype=float)

        down = np.array([0.0, 0.0, -1.0])
        jac_pos = np.zeros((3, self.model.nv))
        jac_rot = np.zeros((3, self.model.nv))

        seeds = [seed.copy()]
        seeds.extend(
            self.rng.uniform(self.lower, self.upper)
            for _ in range(15)
        )

        solutions = []
        best = None

        for seed_index, initial in enumerate(seeds):
            q = np.clip(initial, self.lower, self.upper)
            self.ik_data.qpos[:] = self.data.qpos

            for _ in range(350):
                self.check_operation()
                self.ik_data.qpos[self.qpos_ids] = q
                mujoco.mj_forward(self.model, self.ik_data)

                position = self.ik_data.site_xpos[self.site_id]
                rotation = self.ik_data.site_xmat[
                    self.site_id
                ].reshape(3, 3)
                axis = rotation @ self.approach_local

                position_error = target - position
                axis_error = down - axis
                distance = float(np.linalg.norm(position_error))
                axis_norm = float(np.linalg.norm(axis_error))
                angle = math.degrees(math.acos(float(np.clip(
                    np.dot(axis, down), -1.0, 1.0
                ))))
                score = distance + 0.1 * axis_norm

                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "position_error_mm": distance * 1000,
                        "axis_error_deg": angle,
                    }

                if (
                    distance < cfg.IK_POSITION_TOL
                    and axis_norm < cfg.IK_AXIS_TOL
                ):
                    solutions.append(q.copy())
                    break

                mujoco.mj_jacSite(
                    self.model, self.ik_data,
                    jac_pos, jac_rot, self.site_id,
                )

                # 위치와 접근축 제약을 감쇠 최소제곱법으로 함께 풉니다.
                weight = 0.1
                jac_axis = -skew(axis) @ jac_rot[:, self.dof_ids]
                jacobian = np.vstack([
                    jac_pos[:, self.dof_ids],
                    weight * jac_axis,
                ])
                error = np.concatenate([
                    position_error,
                    weight * axis_error,
                ])

                delta = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + 0.01**2 * np.eye(6),
                    error,
                )
                q = np.clip(
                    q + np.clip(delta, -0.08, 0.08),
                    self.lower,
                    self.upper,
                )

            if solutions and seed_index == 0:
                break

        if not solutions:
            self.last_ik_metrics = best
            raise RuntimeError(
                f"IK 실패: 위치 오차 "
                f"{best['position_error_mm']:.2f}mm, "
                f"방향 오차 {best['axis_error_deg']:.2f}도"
            )

        solution = min(
            solutions,
            key=lambda value: np.linalg.norm(value - seed),
        )

        self.ik_data.qpos[self.qpos_ids] = solution
        mujoco.mj_forward(self.model, self.ik_data)

        axis = (
            self.ik_data.site_xmat[self.site_id].reshape(3, 3)
            @ self.approach_local
        )
        self.last_ik_metrics = {
            "position_error_mm": float(np.linalg.norm(
                target - self.ik_data.site_xpos[self.site_id]
            )) * 1000,
            "axis_error_deg": math.degrees(math.acos(float(np.clip(
                np.dot(axis, down), -1.0, 1.0
            )))),
        }

        return solution

    def wait_reached(self, viewer, position, timeout=cfg.REACH_TIMEOUT):
        """실제 위치·방향·속도가 안정적으로 목표에 도달했는지 검사합니다."""
        deadline = self.data.time + timeout
        stable_since = None

        while True:
            self.check_operation(viewer)
            mujoco.mj_forward(self.model, self.data)

            position_error = float(np.linalg.norm(
                self.data.site_xpos[self.site_id] - position
            ))
            axis = (
                self.data.site_xmat[self.site_id].reshape(3, 3)
                @ self.approach_local
            )
            angle = math.degrees(math.acos(float(np.clip(
                np.dot(axis, [0, 0, -1]), -1.0, 1.0
            ))))
            speed = float(np.max(
                np.abs(self.data.qvel[self.dof_ids])
            ))

            self.last_reach_metrics = {
                "position_error_mm": position_error * 1000,
                "axis_error_deg": angle,
                "joint_speed_rad_s": speed,
            }

            valid = (
                np.isfinite([position_error, angle, speed]).all()
                and position_error <= cfg.REACH_POSITION_TOL
                and angle <= cfg.REACH_ANGLE_TOL_DEG
                and speed <= cfg.REACH_SPEED_TOL
            )

            if valid:
                if stable_since is None:
                    stable_since = self.data.time
                if (
                    self.data.time - stable_since
                    >= cfg.REACH_STABLE_SECONDS
                ):
                    return self.last_reach_metrics.copy()
            else:
                stable_since = None

            if self.data.time >= deadline:
                raise RuntimeError(
                    f"실제 도달 실패: {position_error * 1000:.1f}mm, "
                    f"방향 {angle:.1f}도, 속도 {speed:.3f}rad/s"
                )

            self.step(viewer)

    def plan_path(self, source):
        """모든 작업 구간의 IK를 실제 동작 전에 계산합니다."""
        hover = source.copy()
        hover[2] = cfg.HOVER_Z

        destination = cfg.PLACE.copy()
        destination_hover = destination.copy()
        destination_hover[2] = cfg.HOVER_Z

        points = [
            hover, source, hover,
            destination_hover, destination, destination_hover,
        ]

        # 검증된 손목 방향에 가까운 IK 가지에서 시작합니다.
        seed = np.radians([0, -60, 60, 0, 0])
        plans = []

        for index, end in enumerate(points):
            samples = [end] if index == 0 else [
                points[index - 1] + t * (end - points[index - 1])
                for t in np.linspace(0, 1, 31)[1:]
            ]

            joints = []
            for point in samples:
                seed = self.solve_ik(point, seed)
                joints.append(seed.copy())

            plans.append((joints, end.copy()))

        return plans

    def run_segment(self, viewer, segment, seconds):
        """계획된 관절 경로를 실행하고 실제 도달을 확인합니다."""
        joints, position = segment

        for q in joints:
            self.check_operation(viewer)
            target = self.data.ctrl.copy()
            target[self.arm_ids] = q
            self.move_controls(viewer, target, seconds / len(joints))

        self.wait_reached(viewer, position)

    def evaluate_destination(self):
        """시뮬레이터 정답 상태를 사용해 이동 결과만 평가합니다."""
        actual = self.data.xpos[self.cube_id].copy()
        xy_error = float(np.linalg.norm(
            actual[:2] - cfg.PLACE[:2]
        ))
        z_error = abs(float(actual[2] - cfg.PLACE[2]))

        return {
            "success": (
                xy_error < cfg.FINAL_XY_TOL
                and z_error < cfg.FINAL_Z_TOL
            ),
            "cube_m": actual.tolist(),
            "xy_error_mm": xy_error * 1000,
            "z_error_mm": z_error * 1000,
        }

    def park(self, viewer):
        """다음 관찰에서 큐브를 가리지 않도록 팔을 옆으로 돌립니다."""
        self.stage = "park"
        self.check_operation(viewer)

        target = self.data.ctrl.copy()
        pose = np.radians([0, -60, 60, 0, 0])
        pose[0] = target[self.arm_ids[0]]
        target[self.arm_ids] = pose

        self.move_controls(viewer, target, 2.5)
        target[self.arm_ids[0]] = math.radians(60)
        self.move_controls(viewer, target, 2.0)
        self.hold(viewer, 0.5)

    def execute_pick(self, viewer, source):
        """관찰된 좌표를 기준으로 집기·운반·놓기를 수행합니다."""
        self.check_operation(viewer)

        if self.faulted:
            raise RuntimeError("실패 상태입니다. 먼저 초기화하세요.")

        source = np.asarray(source, dtype=float)
        if source.shape != (3,) or not np.isfinite(source).all():
            raise ValueError("유효한 3차원 파지 좌표가 필요합니다.")

        if (
            np.linalg.norm(source[:2] - cfg.PLACE[:2])
            < cfg.ALREADY_AT_DESTINATION_TOL
        ):
            return {
                "status": "already_at_destination",
                "message": "선택 물체가 이미 목적지에 있습니다.",
            }

        before = self.data.xpos[self.cube_id].copy()
        self.stage = "planning"
        plans = self.plan_path(source)

        self.set_gripper(viewer, cfg.GRIP_OPEN)

        # 단계별 목표를 실제로 달성한 뒤 다음 단계로 진행합니다.
        for index, stage, title, seconds in [
            (0, "approach", "1. 접근", 4.0),
            (1, "descend", "2. 하강", 2.0),
        ]:
            self.stage = stage
            print(title)
            self.run_segment(viewer, plans[index], seconds)

        self.stage = "grasp"
        print("3. 집기")
        self.set_gripper(viewer, cfg.GRIP_CLOSE)

        self.stage = "lift"
        print("4. 들어 올리기")
        self.run_segment(viewer, plans[2], 2.5)

        if self.data.xpos[self.cube_id, 2] < before[2] + 0.025:
            raise RuntimeError("파지 실패: 큐브가 함께 올라오지 않았습니다.")

        for index, stage, title, seconds in [
            (3, "transfer", "5. 목적지로 이동", 3.0),
            (4, "lower", "6. 내려놓기", 2.5),
        ]:
            self.stage = stage
            print(title)
            self.run_segment(viewer, plans[index], seconds)

        self.stage = "release"
        print("7. 놓기")
        self.set_gripper(viewer, cfg.GRIP_OPEN)

        self.stage = "retreat"
        self.run_segment(viewer, plans[5], 2.0)
        self.hold(viewer, 1.0)

        evaluation = self.evaluate_destination()
        return {
            "status": "success" if evaluation["success"] else "failed",
            "source_from_vision_m": source.tolist(),
            "destination_m": cfg.PLACE.tolist(),
            "evaluation": evaluation,
            "message": (
                "가상 큐브 이동 성공"
                if evaluation["success"]
                else "목적지 위치 검증 실패"
            ),
        }