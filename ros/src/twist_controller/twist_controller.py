from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                 accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):

        self.yaw_controller = YawController(wheel_base=wheel_base,
                                            steer_ratio=steer_ratio,
                                            min_speed=0.1,
                                            max_lat_accel=max_lat_accel,
                                            max_steer_angle=max_steer_angle)

        kp = 0.3  # 0.3
        ki = 0.1  # 0.1
        kd = 0.0  # 0.0
        mn = 0.0
        mx = accel_limit

        self.throttle_controller = PID(kp=kp, ki=ki, kd=kd, mn=mn, mx=mx)

        tau = 1.0  # 0.5
        ts = 1.0  # 0.02

        self.throttle_lpf = LowPassFilter(tau=tau, ts=ts)

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.last_vel = 0.
        self.last_time = rospy.get_time()
        self.process_count = 0

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):

        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0., 0., 0.

        self.process_count += 1

        steering = self.yaw_controller.get_steering(linear_velocity=linear_vel,
                                                    angular_velocity=angular_vel,
                                                    current_velocity=current_vel)

        vel_error = linear_vel - current_vel
        self.last_vel = current_vel

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.throttle_controller.step(vel_error, sample_time)
        throttle = self.throttle_lpf.filt(throttle)

        brake = 0

        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400  # N*m - to hold the car in place if we are stopped at a light. Acceleration - 1m/s^2

        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius

        return throttle, brake, steering









