from __future__ import annotations

import time

import numpy as np

import tampura_environments.panda_utils.pb_utils as pbu
from tampura_environments.panda_utils.grasping import get_pregrasp


class ParentBody(object):  # TODO: inherit from Shape?
    def __init__(self, body=None, link=pbu.BASE_LINK, client=None, **kwargs):
        self.body = body
        self.client = client
        self.link = link

    def __iter__(self):
        return iter([self.body, self.link])

    def get_pose(self):
        if self.body is WORLD_BODY:
            return pbu.unit_pose()
        return pbu.get_link_pose(self.body, self.link, client=self.client)

    # TODO: hash & equals by extending tuple
    def __repr__(self):
        return "Parent({})".format(self.body)


DRAW_Z = 1e-2
USE_CONSTRAINTS = True
LEAD_CONTROLLER = True
WORLD_BODY = None


def step_curve(robot, joints, curve, time_step=2e-2, print_freq=None, **kwargs):
    start_time = time.time()
    num_steps = 0
    time_elapsed = 0.0
    last_print = time_elapsed
    for num_steps, (time_elapsed, positions) in enumerate(
        pbu.sample_curve(curve, time_step=time_step)
    ):
        pbu.set_joint_positions(robot, joints, positions, **kwargs)

        if (print_freq is not None) and (print_freq <= (time_elapsed - last_print)):
            print(
                "Step: {} | Sim secs: {:.3f} | Real secs: {:.3f} | Steps/sec {:.3f}".format(
                    num_steps,
                    time_elapsed,
                    pbu.elapsed_time(start_time),
                    num_steps / pbu.elapsed_time(start_time),
                )
            )
            last_print = time_elapsed
        yield positions
    if print_freq is not None:
        print(
            "Simulated {} steps ({:.3f} sim seconds) in {:.3f} real seconds".format(
                num_steps, time_elapsed, pbu.elapsed_time(start_time)
            )
        )


def get_joint_velocities(body, joints, **kwargs):
    return tuple(pbu.get_joint_velocity(body, joint, **kwargs) for joint in joints)


def follow_path(
    body,
    joints,
    path,
    waypoint_tol=1e-2 * np.pi,
    goal_tol=5e-3 * np.pi,
    waypoint_timeout=1.0,
    path_timeout=np.inf,
    lead_step=0.1,
    **kwargs,
):
    dt = pbu.get_time_step(**kwargs)
    handles = []
    steps = 0
    duration = 0.0
    odometry = [
        np.array(pbu.get_joint_positions(body, joints, **kwargs))
    ]  # TODO: plot the comparison with the nominal trajectory
    pbu.control_joints(body, pbu.get_movable_joints(body, **kwargs), **kwargs)
    for num, positions in enumerate(path):
        if duration > path_timeout:
            break
        # start = duration
        is_goal = num == len(path) - 1
        tolerance = goal_tol if is_goal else waypoint_tol

        # TODO: adjust waypoint_timeout based on the duration
        if lead_step is None:
            controller = pbu.joint_controller(
                body,
                joints,
                positions,
                tolerance=tolerance,
                timeout=waypoint_timeout,
                **kwargs,
            )
        else:
            controller = pbu.waypoint_joint_controller(
                body,
                joints,
                positions,
                tolerance=tolerance,
                time_step=lead_step,
                timeout=waypoint_timeout,
                **kwargs,
            )
        for output in controller:
            yield output
            # step_simulation()
            # wait_if_gui()
            # wait_for_duration(10*dt)
            steps += 1
            duration += dt
            odometry.append(
                odometry[-1]
                + dt * np.array(get_joint_velocities(body, joints, **kwargs))
            )
        pbu.remove_handles(handles)


def follow_curve(body, joints, positions_curve, time_step=1e-1, **kwargs):
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/pdControl.py
    control_times = np.append(
        np.arange(positions_curve.x[0], positions_curve.x[-1], step=time_step),
        [positions_curve.x[-1]],
    )
    # TODO: sample_curve
    # velocities_curve = positions_curve.derivative()
    path = [positions_curve(control_time) for control_time in control_times]
    return follow_path(body, joints, path, **kwargs)


class RelativePose(object):
    def __init__(
        self,
        body,
        parent=None,
        parent_state=None,
        relative_pose=None,
        important=False,
        **kwargs,
    ):
        self.body = body
        self.parent = parent
        self.parent_state = parent_state
        if not isinstance(self.body, int):
            self.body = int(str(self.body).split("#")[1])
        if relative_pose is None:
            relative_pose = pbu.multiply(
                pbu.invert(self.get_parent_pose(**kwargs)),
                pbu.get_pose(self.body, **kwargs),
            )
        self.relative_pose = tuple(relative_pose)
        self.important = important

    @property
    def value(self):
        return self.relative_pose

    def ancestors(self):
        if self.parent_state is None:
            return [self.body]
        return self.parent_state.ancestors() + [self.body]

    def get_parent_pose(self, **kwargs):
        if self.parent is WORLD_BODY:
            return pbu.unit_pose()
        if self.parent_state is not None:
            self.parent_state.assign(**kwargs)
        return self.parent.get_pose(**kwargs)

    def get_pose(self, **kwargs):
        return pbu.multiply(self.get_parent_pose(**kwargs), self.relative_pose)

    def assign(self, **kwargs):
        world_pose = self.get_pose(**kwargs)
        pbu.set_pose(self.body, world_pose, **kwargs)
        return world_pose

    def get_attachment(self, **kwargs):
        assert self.parent is not None
        parent_body, parent_link = self.parent
        return pbu.Attachment(
            parent_body, parent_link, self.relative_pose, self.body, **kwargs
        )

    def __repr__(self):
        name = "wp" if self.parent is WORLD_BODY else "rp"
        return "{}{}".format(name, id(self) % 1000)


#######################################################


class Grasp(object):  # RelativePose
    def __init__(self, body, grasp, pregrasp=None, closed_position=0.0, **kwargs):
        # TODO: condition on a gripper (or list valid pairs)
        self.body = body
        self.grasp = grasp
        if pregrasp is None:
            pregrasp = get_pregrasp(grasp)
        self.pregrasp = pregrasp
        self.closed_position = closed_position  # closed_positions

    @property
    def value(self):
        return self.grasp

    @property
    def approach(self):
        return self.pregrasp

    def create_relative_pose(
        self, robot, link=pbu.BASE_LINK, **kwargs
    ):  # create_attachment
        parent = ParentBody(body=robot, link=link, **kwargs)
        return RelativePose(
            self.body, parent=parent, relative_pose=self.grasp, **kwargs
        )

    def create_attachment(self, *args, **kwargs):
        # TODO: create_attachment for a gripper
        relative_pose = self.create_relative_pose(*args, **kwargs)
        return relative_pose.get_attachment()

    def __repr__(self):
        return "g{}".format(id(self) % 1000)


class Conf(object):  # TODO: parent class among Pose, Grasp, and Conf
    # TODO: counter
    def __init__(self, body, joints, positions=None, important=False, **kwargs):
        # TODO: named conf
        self.body = body
        self.joints = joints
        assert positions is not None
        self.positions = tuple(positions)
        self.important = important
        # TODO: parent state?

    @property
    def robot(self):
        return self.body

    @property
    def values(self):
        return self.positions

    def assign(self, **kwargs):
        pbu.set_joint_positions(self.body, self.joints, self.positions, **kwargs)

    def iterate(self):
        yield self

    def __repr__(self):
        return "q{}".format(id(self) % 1000)


class GroupConf(Conf):
    def __init__(self, body, group, *args, **kwargs):
        joints = body.get_group_joints(group, **kwargs)
        super(GroupConf, self).__init__(body, joints, *args, **kwargs)
        self.group = group

    def __repr__(self):
        return "{}q{}".format(self.group[0], id(self) % 1000)


class Command(object):
    @property
    def context_bodies(self):
        return set()

    def iterate(self, state, **kwargs):
        raise NotImplementedError()

    def controller(self, *args, **kwargs):
        raise NotImplementedError()

    def execute(self, controller, *args, **kwargs):
        # raise NotImplementedError()
        return True

    def to_lisdf(self):
        raise NotImplementedError


class Trajectory(Command):
    def __init__(
        self,
        body,
        joints,
        path,
        velocity_scale=1.0,
        contact_links=[],
        time_after_contact=np.inf,
        contexts=[],
        **kwargs,
    ):
        self.body = body
        self.joints = joints
        self.path = tuple(path)  # waypoints_from_path
        self.velocity_scale = velocity_scale
        self.contact_links = tuple(contact_links)
        self.time_after_contact = time_after_contact
        self.contexts = tuple(contexts)
        # self.kwargs = dict(kwargs) # TODO: doesn't save unpacked values

    @property
    def robot(self):
        return self.body

    @property
    def context_bodies(self):
        return {self.body} | {
            context.body for context in self.contexts
        }  # TODO: ancestors

    def conf(self, positions):
        return Conf(self.body, self.joints, positions=positions)

    def first(self):
        return self.conf(self.path[0])

    def last(self):
        return self.conf(self.path[-1])

    def reverse(self):
        return self.__class__(
            self.body,
            self.joints,
            self.path[::-1],
            velocity_scale=self.velocity_scale,
            contact_links=self.contact_links,
            time_after_contact=self.time_after_contact,
            contexts=self.contexts,
        )

    def adjust_path(self, **kwargs):
        current_positions = pbu.get_joint_positions(
            self.body, self.joints, **kwargs
        )  # Important for adjust_path
        return pbu.adjust_path(
            self.body, self.joints, [current_positions] + list(self.path), **kwargs
        )  # Accounts for the wrap around

    def compute_waypoints(self, **kwargs):
        return pbu.waypoints_from_path(
            pbu.adjust_path(self.body, self.joints, self.path, **kwargs)
        )

    def compute_curve(self, **kwargs):
        path = self.adjust_path(**kwargs)
        positions_curve = pbu.interpolate_path(self.body, self.joints, path, **kwargs)
        return positions_curve

    def iterate(self, state, teleport=False, **kwargs):
        if teleport:
            pbu.set_joint_positions(self.body, self.joints, self.path[-1], **kwargs)
            return self.path[-1]
        else:
            return step_curve(
                self.body, self.joints, self.compute_curve(**kwargs), **kwargs
            )

    def __repr__(self):
        return "t{}".format(id(self) % 1000)


class CaptureImage(Command):
    def __init__(self, robot=None, captured_image=None, **kwargs):
        self.robot = robot
        self.captured_image = captured_image

    def iterate(self, state, **kwargs):
        self.captured_image = self.robot.get_image()
        return pbu.empty_sequence()


class GroupTrajectory(Trajectory):
    def __init__(self, body, group, path, *args, **kwargs):
        # TODO: rename body to robot
        joints = body.get_group_joints(group, **kwargs)
        super(GroupTrajectory, self).__init__(body, joints, path, *args, **kwargs)
        self.group = group

    def reverse(self, **kwargs):
        return self.__class__(
            self.body,
            self.group,
            self.path[::-1],
            velocity_scale=self.velocity_scale,
            contact_links=self.contact_links,
            time_after_contact=self.time_after_contact,
            contexts=self.contexts,
            **kwargs,
        )

    def __repr__(self):
        return "{}t{}".format(self.group[0], id(self) % 1000)
