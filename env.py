"""
Project: Adaptive longitudinal velocity and look-ahead distance adjustment of pure pursuit.
Description: Employ the DDPG algorithm to lean how to control the speed and look-ahead distance of pure pursuit for imporve the path tracking performance.
Modified by: Pang.
License: MIT License
Original Author: idreesshaikh
Original project URL: https://github.com/idreesshaikh/Autonomous-Driving-in-Carla-using-Deep-Reinforcement-Learning

[version3]What is new:    1)这个版本将前视平均曲率、加速度加入状态空间。
                          2)状态空间: [self.mean_curvature, self.acceleration, self.throttle, self.kmh, normalized_distance_from_center, normalized_angle]
                
[version4]20240429 修改策略，低速不结束，只有超时才结束.
[version5]20240429 意识到得在实时模式而非同步模式下训练。在同步模式下训练出来的agent在实时模式下表现抽象。
                    将航向角标准化改成了10度(奖励过于严格,训练不出来,改回20)
                    将jerk改成了横向jerk惩罚
                    将未来曲率的horizon改成了3
                    将lookahead的调整范围改为了[1.2, 6]
[version5-1]20240505 减小道路中心偏离最大值，重新训练。
            20240507 增加最大加速度限制，重新训练。

[version5-2]20240508 平均曲率扩大十倍,让其在区间[0,1]中。 
                     agent改回了DDPG
                     修正look-ahead的选取方式,固定一个l,学习一个增益k,让其在[l-k,l+k]中变化,充分利用激活函数tanh的特性,使其变化与油门相似,尝试学习。
[version6] 20240508 增加了SACAgent,发现训练稳定,比ddpg容易收敛.
                    修改了奖励函数,让agent在直路上偏重于跟随限速行驶,在弯道处偏重于贴合路径.
                    增加了accel_factor因子来惩罚加速度过大的行为。
           20240513 增加了jerk_factor来约束jerk过大的行为。
           20240514 优化了accel的奖励并增加了可视化加速度向量的函数
                    踩刹车踩得太死了，应该设置一个刹车上限，jerk都是因为刹车产生的
                    将平均曲率修改回了原来的值，防止学习过于注重曲率。
"""
# 20240411 平均曲率差 完成 奖励函数没写。


# 放宽边界条件
# 增加一个时间惩罚，每运行一次step都有一个固定的成本(不行，训练不出来，)


import carla
import time
import random
import numpy as np
import pygame
import math
from waypoints_utils import load_wp_curve, load_waypoints, draw_waypoints, find_lookahead_waypoint, curvature_yaw_diff, draw_acceleration_vector
from parameters import *
import csv
from PID import PIDController


class CarEnv():
    '''
    A-to-B Navication task environment with predefined path. 
    '''
    STEER_AMT = 0.3 # 转向最大值
    def __init__(self, checkpoint_frequency=100, sync=False, continuous_action_space=True, render=True, train=True, 
                 town='Town05', pathfile='town5_waypoints', pid=None, pp=False, determined_speed=None):
        self.client = carla.Client(HOST, PORT)
        self.client.set_timeout(5.0)
        self.client.load_world(town)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.collision_list = []
        self.actor_list = []
        self.vehicle_bp = self.blueprint_library.filter("model3")[0]
        self.im_width = IM_WIDTH # pygame窗口大小
        self.im_height = IM_HEIGHT
        self.timesteps = 0 # 记录与环境交互的步数
        self.current_waypoint_index = 0
        self.route_waypoints = None # 道路行点列表
        self.continuous_action_space = continuous_action_space # 是否为连续动作空间的flag
        self.pathfile = pathfile
        #=====里程计=====reset()函数中会再次初始化
        self.kmh = 0 
        self.max_kmh = 30 # 限速
        self.target_kmh = 25
        self.min_kmh = 15
        self.max_acceleration = 4.0 #[version5-1]
        #=====其余状态信息=====
        self.max_distance_from_center = 0
        self.angle = 0
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.mean_curvature = 0 # [version3]平均曲率
        #=====================
        self.fresh_start = True
        self.distance_from_center = None
        self.original_setting = None
        self.sync = sync # 同步模式flag
        self.render = render # 渲染模式
        self.train = train # 训练模式
        self.pid_controller = pid if isinstance(pid, PIDController) else PIDController(kp=0.8, ki=0.0, kd=0.1, dt=0.05) 
        #None # 使用PID控制运行
        self.determined_speed = determined_speed if isinstance(determined_speed, PIDController) else PIDController(kp=0.8, ki=0.0, kd=0.1, dt=0.05) 
        #None# 是否定速
        self.pp = pp # 训练混合pp控制器
        self.checkpoint_frequency = checkpoint_frequency
        self.checkpoint_waypoint_index = 0
        self.target_waypoint = None # 用于根据预瞄距离选取目标路点
        self.original_setting = self.world.get_settings()
        #========plot path======
        self._path = list()
        self._vpath = list()
        self.info = dict()
        
        self.obstacle_trigger_distance = 18.0  # 小于此值切换为 RL 控制
        self.obstacle_safe_distance = 2.0     # 大于此值切换回 PP 控制


        if self.sync:
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.05 #20FPS 50ms时间步长
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
        if not self.render:
            settings = self.world.get_settings()
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)

    def reset(self):
        # try:
                            
            self.collision_hist = []
            self.lane_invasion_hist = []
            self.actor_list = []

            # 创建pygame窗口
            # self.display = pygame.display.set_mode(
            #     (800, 600),
            #     pygame.SWSURFACE | pygame.DOUBLEBUF)

            # 车辆生成
            self.spawn_point = random.choice(self.map.get_spawn_points())
            self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)
            self.actor_list.append(self.vehicle)

            #生成静态障碍物
            obstacle_transform = carla.Transform(
            carla.Location(x=31.5, y=145.5, z=0.1),
            carla.Rotation(yaw=270)
            )
            obstacle_bp = self.blueprint_library.filter("vehicle.tesla.model3")[0]
            self.static_obstacle = self.world.try_spawn_actor(obstacle_bp, obstacle_transform)

            if self.static_obstacle:
                self.static_obstacle.set_simulate_physics(False)
                self.actor_list.append(self.static_obstacle)

            # 相机蓝图与参数
            self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
            self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
            self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
            self.rgb_cam.set_attribute("fov", f"110")

            #相机附着到车辆
            cam_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
            self.cam_sensor = self.world.spawn_actor(self.rgb_cam, cam_transform, attach_to=self.vehicle)
            self.actor_list.append(self.cam_sensor)
            self.cam_sensor.listen(lambda data: self.process_img(data))

            #碰撞传感器蓝图与附着到车辆
            self.col_sensor_bp = self.blueprint_library.find("sensor.other.collision")
            col_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
            self.col_sensor = self.world.spawn_actor(self.col_sensor_bp, col_transform, attach_to=self.vehicle)
            self.actor_list.append(self.col_sensor)
            self.col_sensor.listen(lambda event: self.collision_data(event))

            # 添加 Lane Invasion Sensor
            self.lane_sensor_bp = self.blueprint_library.find("sensor.other.lane_invasion")
            lane_transform = carla.Transform()
            self.lane_sensor = self.world.spawn_actor(self.lane_sensor_bp, lane_transform, attach_to=self.vehicle)
            self.actor_list.append(self.lane_sensor)
            # 监听函数：只记录“实线”压线
            self.lane_sensor.listen(lambda event: self._record_lane_invasion(event))

            #初始化
            self.timesteps = 0
            self.rotation = self.vehicle.get_transform().rotation.yaw
            self.previous_location = self.vehicle.get_location()
            self.acceleration = self.vehicle.get_acceleration()
            self.velocity = self.vehicle.get_velocity()
            self.max_kmh = L_MAX_SPEED
            self.target_kmh = L_TARGET_SPEED
            self.min_kmh = L_MIN_SPEED
            self.max_distance_from_center = MAX_DISTANCE_FROM_CENTER 
            self.throttle = float(0.0)
            self.previous_steer = float(0.0)
            self.kmh = float(0.0)
            self.angle = float(0.0)
            self.distance_from_center = float(0.0)
            self.center_lane_deviation = 0.0
            self.total_distance = 200
            self.checkpoint_waypoint_index =0
            # self.curvatures = list() #[version3]
            self.mean_curvature = 0 #[version3]平均曲率
            self.next_curvature = 0
            self.lookahead = 0 #[version3] 前视距离
            self.max_acceleration = 4.0 #[version5-1] 最大加速度限制
            self.max_jerk = 2 #[version6] 最大jerk限制
            # initializion  of obstacle information
            self.distance_to_obstacle = 100.0
            self.angle_to_obstacle = 0.0
            #avoidance_mode
            self.avoidance_mode = False
            

            self.last_y = self.vehicle.get_location().y

 
            


            # 用来监测jerk惩罚和角速度惩罚的
            self.all_jerk = 0
            self.all_rotation_diff = 0


            #第一次运行， 生成路点
            if self.fresh_start:
                self.current_waypoint_index = 0
                self.route_waypoints = list()
                # 从车辆位置迭代生成航点的代码（不使用）
                # self.waypoint = self.map.get_waypoint(self.vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
                # current_waypoint = self.waypoint
                # self.route_waypoints.append(current_waypoint)
                # for x in range(self.total_distance):
                #     next_waypoint = current_waypoint.next(0.5)[0]
                #     self.route_waypoints.append(next_waypoint)
                #     print(current_waypoint.id)
                #     current_waypoint = next_waypoint
                if not self.route_waypoints:
                    self.route_waypoints, self.curvatures = load_wp_curve(self.pathfile)
                    draw_waypoints(self.world, self.route_waypoints, z=0, lifetime=20)
                    waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                    car_location = carla.Location(x=waypoint[0], y=waypoint[1], z=0)
                    _transform = self.map.get_waypoint(car_location)
                    car_transform = _transform.transform
                    self.vehicle.set_transform(car_transform)

            else:
                waypoint = self.route_waypoints[self.checkpoint_waypoint_index % len(self.route_waypoints)]
                car_location = carla.Location(x=waypoint[0], y=waypoint[1], z=0)
                _transform = self.map.get_waypoint(car_location)
                car_transform = _transform.transform
                self.vehicle.set_transform(car_transform)
                self.current_waypoint_index = self.checkpoint_waypoint_index
    
            
            #车辆控制初始化
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
            time.sleep(1)

            #构建返回的状态信息
            acceleration = math.sqrt(self.acceleration.x**2 + self.acceleration.y**2 + self.acceleration.z**2)
            #state for obstacle avoidance training
            self.state = np.array([self.distance_to_obstacle, self.angle_to_obstacle,acceleration, self.kmh, self.distance_from_center, self.angle])
            #state for pp training
            #self.state = np.array([self.mean_curvature, acceleration, self.kmh, self.distance_from_center, self.angle]) 
            #碰撞事件列表
            self.collision_hist = []
            # 压线事件列表
            self.lane_invasion_hist = []
            #episode开始时间
            self.episode_start_time = time.time()

            #返回状态空间
            return self.state, self.route_waypoints
        # except:
        #     print("reset function error")
        #     if self.actor_list:
        #         for actor in self.actor_list:
        #             actor.destroy()
        #     if self.sync:
        #         self.world.apply_settings(self.original_setting)
        #     if not self.render:
        #         self.world.apply_settings(self.original_setting)

    
    #与环境交互的函数
    def step(self, action):
        # try:
            if self.sync:
                self.world.tick() # 同步模式下使carla运行一个步长
            # 设置监视器俯视
            spectator = self.world.get_spectator()
            spectator_transform = self.vehicle.get_transform()
            spectator.set_transform(carla.Transform(spectator_transform.location + carla.Location(z=20),
                                                    carla.Rotation(pitch=-90)))
            self.fresh_start = False
            self.timesteps += 1

            # 当前时间步奖励置零
            reward = 0
            throttle_control = 0.3
            
            """
            # 动作解包，前视距离只有大于0时才是有效的
            # print("动作内容：", action)
            action_1 = action[0][0] 
            action_2 = action[0][1]
            lookahead = action_1
            lookahead = float((lookahead+1.0)/2)
            lookahead = 1 + min(max(lookahead, 0), 1.0) * 6
            # lookahead = min(max(lookahead, 0), 1.0) * 7
            # lookahead = float(1.5 + 1.5*lookahead)
            throttle = float(action_2)
            """
            
            # 获取当前车辆朝向和位置信息
            self.location = self.vehicle.get_location()
            self._vpath = [self.location.x,self.location.y]
            vehicle_location = np.array(self._vpath)
            rotation = self.vehicle.get_transform().rotation.yaw
            
            #get the information between obstacle and vehicle
            obstacle_loaction = np.array([self.static_obstacle.get_location().x,self.static_obstacle.get_location().y])
            obstacle_distance = np.linalg.norm(obstacle_loaction - vehicle_location)
            dx, dy = obstacle_loaction - vehicle_location
            obstacle_angle = np.arctan2(dy, dx) - np.deg2rad(self.vehicle.get_transform().rotation.yaw)
            obstacle_angle = (obstacle_angle + np.pi) % (2 * np.pi) - np.pi

            # 获取当前车辆航点信息，通过find_lookahead_waypoint函数找到目标预瞄点
            #self.current_waypoint = self.map.get_waypoint(self.location, project_to_road=True, lane_type=(carla.LaneType.Driving))
            #self.target_waypoint = find_lookahead_waypoint(vehicle_location,self.current_waypoint_index, lookahead, self.route_waypoints)
            v_transform = self.vehicle.get_transform()
            

            #print(f"[DEBUG] mode: {'RL' if self.avoidance_mode else 'PID'}, obstacle_distance: {obstacle_distance:.2f}, action: {action}")

            if not self.avoidance_mode and obstacle_distance < self.obstacle_trigger_distance:
                print("[切换] 避障模式激活！")
                self.avoidance_mode = True
            # elif self.avoidance_mode and obstacle_distance > self.obstacle_safe_distance:
            #     print("[切换] 恢复路径跟踪模式！")
            #     self.avoidance_mode = False

            if self.avoidance_mode:
                # 使用 RL 控制器
                throttle = float(action[0][0]) # 或 action[0] if flatten
                steer = float(action[0][1])
                self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
            else:
                # 使用 PID 控制器
                # 1. throttle 控制（可使用 PID 或固定值）
                speed_value = self.kmh # 当前速度 km/h（需自行转换为 m/s）
                throttle_control = self.determined_speed.update(round(speed_value, 2))  
                # 2. steer 控制（基于横向误差）
                throttle = self.determined_speed.update(round(speed_value, 2))
                deviation = self.distance_from_center * self.check_deviation_left_right(vehicle_location)
                steer = self.pid_controller.update(round(deviation, 2))  
                # 3. 应用控制
                self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

            
            """
            # 使用pid控制器
            if self.pid_controller:
                throttle_control = (self.throttle*0.8 + throttle*0.2)
                value = self.distance_from_center * self.check_deviation_left_right(vehicle_location)
                steer_pid = self.pid_controller.update(round(value, 2))
                self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_control, steer=steer_pid))
            # 使用PP作为控制器
            elif self.pp:
                lookahead = action_1
                speed_value = self.kmh
                throttle_control = self.determined_speed.update(round(speed_value,2))
                self.target_waypoint = find_lookahead_waypoint(vehicle_location,self.current_waypoint_index, lookahead, self.route_waypoints)
                steer = self.pure_pursuit(self.target_waypoint, v_transform)
                if throttle_control >= 0:
                    self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_control, steer=steer))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=steer, brake=-throttle_control))
            # rl-based混合控制器
            else:
                steer = self.pure_pursuit(self.target_waypoint, v_transform)
                throttle_control = (self.throttle*0.8 + throttle*0.2)
                if throttle_control >= 0:
                    self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_control, steer=steer))
                else:
                    self.vehicle.apply_control(carla.VehicleControl(steer=steer, brake=-throttle_control))

            """
            
            # 保存上一帧油门信息
            self.throttle = throttle_control

            # 获取里程计信息
            velocity = self.vehicle.get_velocity()
            acceleration = self.vehicle.get_acceleration()
            draw_acceleration_vector(world=self.world, vehicle=self.vehicle, accel=acceleration)
            # rotation_diff = math.sqrt((rotation - self.rotation)**2)
            # rotation_diff = abs(rotation_diff)
            # rotation_diff = rotation_diff * (math.pi/180)
            # self.rotation = rotation
            yaw_velocity = self.vehicle.get_angular_velocity().z


            self.kmh = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            foward_jerk = acceleration.x - self.acceleration.x
            foward_jerk = abs(foward_jerk)/0.05

            long_jerk, lat_jerk = self.get_local_jerk(acceleration, v_transform)

            self.acceleration = acceleration
            # lat_acc = abs(acceleration.y)
            lat_acc = self.get_lat_acc(velocity)
            acceleration = math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)



            # 检查车辆是否通过最近的航点，通过了则更新next_waypoint
            
            #self.current_waypoint_index = self.check_vehicle_current_index(vehicle_location)

            # 计算横向偏差
            self.current_waypoint = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
            self.next_waypoint = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
            self.distance_from_center = self.distance_to_line(self.current_waypoint, self.next_waypoint, vehicle_location)
            self.center_lane_deviation += self.distance_from_center

            #朝向对齐程度
            vh_yaw = self.correct_yaw(self.vehicle.get_transform().rotation.yaw)
            delta = np.array(self.next_waypoint) - np.array(self.current_waypoint)
            wp_yaw = np.arctan2(delta[1], delta[0]) * 180.0 / np.pi # 转换为角度（degree）
            wp_yaw = self.correct_yaw(wp_yaw)
            #wp_yaw = self.correct_yaw(self.next_waypoint.rotation.yaw)  
            cos_yaw_diff = np.cos((vh_yaw - wp_yaw) * np.pi / 180.0)

            #车辆与参考路径点的距离
            vehicle_location = self.vehicle.get_location()
            wp_location = carla.Location(x=self.next_waypoint[0], y=self.next_waypoint[1], z=0)
            dist = np.linalg.norm(np.array([vehicle_location.x - wp_location.x,
                                            vehicle_location.y - wp_location.y]))
            

            #纵向移动距离
            current_y = self.vehicle.get_location().y
            traveled = current_y - self.last_y
            self.last_y = current_y  # 更新


            # 计算角度偏差
            v_forward = self.vector(self.vehicle.get_velocity())
            wp_forward = self.next_waypoint - self.current_waypoint
            self.angle = self.angle_diff(v_forward, wp_forward)

            # [version3]计算当前waypoint往前进方向20个点的平均曲率以及其变化率
            if self.current_waypoint_index < len(self.route_waypoints) - 22:
                mean_curvature = np.mean(self.curvatures[self.current_waypoint_index:self.current_waypoint_index + 20]) 
                next_curvature = self.curvatures[self.current_waypoint_index + 5]
            else: 
                mean_curvature = np.mean(self.curvatures[self.current_waypoint_index:])
                next_curvature = mean_curvature
            mean_curvature = abs(mean_curvature) if mean_curvature else 0
            if np.isnan(mean_curvature):
                mean_curvature = 0
            if np.isnan(next_curvature):
                next_curvature = 0
            self.mean_curvature = mean_curvature
            self.next_curvature = next_curvature

            # [version3]计算lookahead变化率
            #lookahead_diff = lookahead - self.lookahead
            #self.lookahead = lookahead

            '''
            这一部分为奖励函数
            需要根据跟踪的路径进行设计
            '''
            # 非第一次运行，将当前路点idx置为通过的检查点的idx
            if not self.fresh_start:
                if self.checkpoint_frequency is not None:
                    self.checkpoint_waypoint_index = (self.current_waypoint_index // self.checkpoint_frequency) * self.checkpoint_frequency
            done = False
            

            if len(self.collision_hist) !=0:
                print("碰撞结束")
                done = True
                reward = -100
            elif len(self.lane_invasion_hist) != 0:
                print("压线结束")
                done = True
                reward = -100
            # elif self.distance_from_center > self.max_distance_from_center:
            #     print("偏离结束", self.distance_from_center)
            #     done = True
            #     reward = -500
            elif self.episode_start_time + 10 < time.time() and self.kmh < 3:
                print("低速结束", self.kmh)
                done = True
                reward = -500
            elif self.kmh > self.max_kmh:
                print("超速结束", self.kmh)
                done = True
                reward = -500
            
           
            if self.timesteps >= 5000 and self.train:
                print("超时结束")
                done = True
                reward = -self.timesteps
            elif self.current_waypoint_index >= len(self.route_waypoints)-3:
                print("完成结束")
                done = True
                self.fresh_start = True
                reward = 500
                self.checkpoint_waypoint_index = 0
                if self.checkpoint_frequency is not None:
                    if self.checkpoint_frequency < self.total_distance//2:
                        self.checkpoint_frequency += 2
                    else:
                        self.checkpoint_frequency = None
                        self.checkpoint_waypoint_index = 0
            '''
            计算奖励
            Calculate the reward for each time step
            '''
            if not done:
                # reward = self.curvature_reward2(accel=acceleration)
                reward = self.obstacle_reward(cos_yaw_diff, dist, traveled,long_jerk, lat_jerk, lat_acc)
                if throttle_control < 0:
                    reward += throttle_control
                # reward = self.original_reward()
                # reward = self.conventional_reward(accel=lat_acc)
                # reward = self.case_conventional_reward(accel=lat_acc)

            
            # 速度，偏离，角度标准化
            normalized_kmh = self.kmh / self.target_kmh
            normalized_distance_from_center = self.distance_from_center / self.max_distance_from_center
            normalized_angle = self.angle / np.deg2rad(20)
            normalized_accel = acceleration / self.max_acceleration
            #state for obstacle avoidance training
            self.state = np.array([obstacle_distance, obstacle_angle,normalized_accel, normalized_kmh, normalized_distance_from_center, normalized_angle])
            #state for pp training
            # self.state = np.array([mean_curvature, normalized_accel, normalized_kmh, normalized_distance_from_center, normalized_angle])
            self.center_lane_deviation = self.center_lane_deviation / self.timesteps

            # 调试信息
            self.info = {
                "reward":reward,
                "speed":self.kmh, 
                "throttle":throttle_control, 
                #"lookahead":lookahead, 
                "timesteps":self.timesteps, 
                "v_yaw":yaw_velocity, 
                "lat_jerk":lat_jerk, 
                "path_covered":self.current_waypoint_index/len(self.route_waypoints), 
                "ave_deviation":self.center_lane_deviation, 
                "cur_deviation":self.distance_from_center, 
                "lat_accel":lat_acc, 
                "vehicle_location":self._vpath,
                "deviation":self.distance_from_center
            }
            if done:
    
                for actor in self.actor_list:
                    actor.destroy()
                self.actor_list = None

            return self.state, reward, done, self.info

        # 以下代码由于调试暂不使用
        # except:
        #     self.destroy_env()
    

#===============================================
# 其他函数
#===============================================
    def get_lat_acc(self, velocity):
        # 计算时间间隔
        dt = 0.05

        # 计算速度变化量
        dv_x = (velocity.x - self.velocity.x) / dt
        dv_y = (velocity.y - self.velocity.y) / dt

        # 计算当前速度大小
        speed = np.sqrt(velocity.x**2 + velocity.y**2)

        if speed > 0:
            # 计算横向加速度
            lateral_accel = (velocity.x * dv_y - velocity.y * dv_x) / speed
        else:
            lateral_accel = 0

        # 更新速度和时间
        self.velocity = velocity

        return lateral_accel
    
    def get_local_jerk(self, accel, v_transform):
        """计算车辆局部坐标系下的 Jerk"""

        dt = 0.05

        # 获取车辆坐标系方向
        transform = v_transform
        forward = transform.get_forward_vector()
        right = transform.get_right_vector()

        # 计算纵向 & 横向加速度
        accel_long = accel.x * forward.x + accel.y * forward.y
        accel_lat = accel.x * right.x + accel.y * right.y

        prev_accel_long = self.acceleration.x * forward.x + self.acceleration.y * forward.y
        prev_accel_lat = self.acceleration.x * right.x + self.acceleration.y * right.y

        # 计算 Jerk
        jerk_long = (accel_long - prev_accel_long) / dt
        jerk_lat = (accel_lat - prev_accel_lat) / dt


        return jerk_long, jerk_lat
    def vector(self, v):
        '''
        将v转换为vector
        '''
        if isinstance(v, carla.Location) or isinstance(v, carla.Vector3D):
            return np.array([v.x, v.y, v.z])
        elif isinstance(v, carla.Rotation):
            return np.array([v.pitch, v.yaw, v.roll])
    
    # 计算p3到p1,p2连线的距离
    def distance_to_line(self, p1, p2, p3):
        '''
        计算P3到P1-P2连线的距离
        '''
        _cross = np.linalg.norm(np.cross(p1 - p2, p2 - p3))
        _norm = np.linalg.norm(p1 - p2)
        if np.isclose(_norm, 0):
            return np.linalg.norm(p3 - p1)
        return _cross / _norm

    def collision_data(self, event):
        '''
        碰撞传感器回调函数，往碰撞历史数组里添加事件
        '''
        self.collision_hist.append(event)
    

    def _record_lane_invasion(self, event):
        for marking in event.crossed_lane_markings:
            if marking.type == carla.LaneMarkingType.Solid:
                self.lane_invasion_hist.append(event)
                print("[警告] 实线越线")
                break  # 一旦有实线就记录一次

    #使用pygame监视
    def process_img(self, image, surface=None):
        if image is not None:
            image_data = np.array(image.raw_data)
            image_data = image_data.reshape((self.im_height, self.im_width, 4))
            image_data = image_data[:, :, :3]  # 去除alpha通道
            image_data = image_data[:, :, ::-1]
            image_surface = pygame.surfarray.make_surface(image_data.swapaxes(0, 1))
            if surface:
                surface.blit(image_surface, (0,0))
            self.front_camera = image_data

    #角度差值计算
    def angle_diff(self, v0, v1):
        '''
        计算向量v0与v1的角度, 返回弧度
        '''
        angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v0[1], v0[0])
        if angle > np.pi: angle -= 2 * np.pi
        elif angle <= -np.pi: angle += 2 * np.pi
        return angle

    def destroy_env(self) -> None:
        if self.actor_list is not None:
            for actor in self.actor_list:
                actor.destroy()
        if self.sync:
            # settings = self.world.get_settings()
            # settings.synchronous_mode = False
            # self.world.apply_settings(settings)
            self.world.apply_settings(self.original_setting)

    def vector3d_diff(self, v0, v1):

        result = math.sqrt((v0.x-v1.x)**2 + (v0.y-v1.y)**2 + (v0.z-v1.z)**2)
        return result
    

    def pure_pursuit(self, tar_location, v_transform):
        '''
        纯跟踪
        '''
        L = 2.396
        yaw = v_transform.rotation.yaw * (math.pi / 180)
        x = v_transform.location.x - L / 2 * math.cos(yaw)
        y = v_transform.location.y - L / 2 * math.sin(yaw)
        dx = tar_location[0] - x
        dy = tar_location[1] - y
        ld = math.sqrt(dx ** 2 + dy ** 2)
        # ld = L
        alpha = math.atan2(dy, dx) - yaw
        delta = math.atan(2 * math.sin(alpha) * L / ld) * 180 / math.pi
        steer = delta/90
        if steer > 1:
            steer = 1
        elif steer < -1:
            steer = -1
        return steer

    def draw_location(self, world):
        '''获取vehicle位置然后画个点来显示轨迹。'''
        pass

    def save_return_log(self, return_list, filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['waypoint', 'curvature'])
            for i, episode_return in enumerate(return_list):
                writer.writerow([i+1, episode_return])


    def check_vehicle_current_index(self, vehicle_location):
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            next_waypoint_index = waypoint_index + 1
            _waypoint = self.route_waypoints[waypoint_index % len(self.route_waypoints)]
            waypoint_ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(waypoint_ - _waypoint, vehicle_location - _waypoint)
            if dot > 0 :
                waypoint_index += 1
            else:
                break
        return waypoint_index
    
    def check_deviation_left_right(self, vehicle_location):
        waypoint_index = self.current_waypoint_index
        next_waypoint_index = waypoint_index + 1
        _waypoint = self.route_waypoints[waypoint_index % len(self.route_waypoints)]
        waypoint_ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
        cros = np.cross(waypoint_ - _waypoint, vehicle_location - _waypoint)
        if cros > 0 :
            return 1
        return -1

    def original_reward(self):
        centering_factor = max((1.0 - self.distance_from_center / self.max_distance_from_center), 0.0)
        angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)


        if self.kmh < self.min_kmh:
            # reward = (self.kmh / self.target_kmh)
            reward = (self.kmh / self.target_kmh) * centering_factor * angle_factor
            # 超速奖励
        elif self.kmh > self.target_kmh:
            reward = max(1.0 - (self.kmh - self.target_kmh) / (self.max_kmh - self.target_kmh), 0.0) * centering_factor * angle_factor
        # 常速奖励
        else:
            reward = 1.0 * centering_factor * angle_factor
        
        return reward
    
    
    def obstacle_reward(self, cos_yaw_diff, dist, traveled, jerk_long, jerk_lat, lat_acc,
                     lambda_1=1.0, lambda_2=1.0, lambda_5=0.5, lambda_jerk=0.05, lambda_acc=0.1):
        """
        综合奖励函数（考虑轨迹对齐、距离、碰撞、车道线、行驶距离、jerk、横向加速度）
        """
        jerk_penalty = lambda_jerk * (abs(jerk_long) + abs(jerk_lat))
        acc_penalty = lambda_acc * abs(lat_acc)  
        reward = (lambda_1 * cos_yaw_diff) \
            - (lambda_2 * dist) \
            + (lambda_5 * traveled) \
            - jerk_penalty \
            - acc_penalty  
        
        return reward
    
    def curvature_reward(self, accel):
        centering_factor = max((1.0 - self.distance_from_center / self.max_distance_from_center), 0.0)
        angle_factor = max(1.0 - abs(self.angle / np.deg2rad(20)), 0.0)
        accel_factor = max((1.0 - abs(accel / self.max_acceleration)), 0.0)

        if self.mean_curvature > 0.01:
            reward = 1.0 * centering_factor * angle_factor * accel_factor
        elif self.kmh < self.min_kmh:
            # reward = (self.kmh / self.target_kmh)
            reward = (self.kmh / self.target_kmh) * centering_factor * angle_factor
            # 超速奖励
        elif self.kmh > self.target_kmh:
            reward = max(1.0 - (self.kmh - self.target_kmh) / (self.max_kmh - self.target_kmh), 0.0) * centering_factor * angle_factor
        # 常速奖励
        else:
            reward = 1.0 * centering_factor * angle_factor
        
        return reward
    
    def curvature_reward2(self, accel):
        ego_curve = abs(self.curvatures[self.current_waypoint_index])
        centering_factor = 1.0 - self.distance_from_center / self.max_distance_from_center
        angle_factor = 1.0 - abs(self.angle / np.deg2rad(20))
        accel_factor = 1.0 - accel / self.max_acceleration
        speed_target_factor = 1.0 - self.kmh / self.target_kmh
        speed_min_factor = 1.0 - self.kmh / self.min_kmh
        is_turning_threshold = 0.01
        if ego_curve > is_turning_threshold:
            reward = centering_factor + angle_factor + accel_factor + speed_min_factor
        elif self.mean_curvature - ego_curve > 0:
            reward = accel_factor + angle_factor + centering_factor + speed_min_factor
        elif self.mean_curvature - ego_curve <= 0:
            reward = speed_target_factor + angle_factor + centering_factor + 1
        else:
            reward = speed_target_factor + angle_factor + centering_factor + 1
        return reward 
    
    def case_conventional_reward(self, accel):
        rw = { 
                # 该字典用于设置各个奖励项的权重
                "speed":-1,
                "deviation":-1,
                "angle":-5,
                "accel":-1
            }
        turn_rw = { 
                # 该字典用于设置各个奖励项的权重
                "speed":-1,
                "deviation":-5,
                "angle":-1,
                "accel":-5
            }
        speed_factor = abs(self.kmh - self.target_kmh)
        deviation_factor = abs(self.distance_from_center)
        angle_factor = abs(self.angle)
        if self.next_curvature > 0.01:
            reward = turn_rw["speed"] * speed_factor + turn_rw["deviation"]*deviation_factor + turn_rw["angle"]*angle_factor + turn_rw["accel"]*accel
        else:
            reward = rw["speed"] * speed_factor + rw["deviation"]*deviation_factor + rw["angle"]*angle_factor + rw["accel"]*accel
        reward -= 0.1
        return reward
    
    def conventional_reward(self, accel):
        rw = { 
                # 该字典用于设置各个奖励项的权重
                "speed":-5,
                "deviation":-1,
                "angle":-5,
                "accel":-1
            }
        speed_factor = abs(self.kmh - self.target_kmh)
        deviation_factor = abs(self.distance_from_center)
        angle_factor = abs(self.angle)
        reward = rw["speed"] * speed_factor + rw["deviation"]*deviation_factor + rw["angle"]*angle_factor + rw["accel"]*accel
        reward -= 0.1
        return reward
    
    def correct_yaw(self,yaw):
        yaw = yaw % 360
        if yaw > 180:
            yaw -= 360
        return yaw

    
