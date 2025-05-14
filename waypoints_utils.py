# 20240322 Created
'''
Discription: Some utils about carla.Waypoints
Author: Pang.
'''
import numpy as np
import pickle
import carla
import math
import os
import pandas as pd

def save_waypoints(wpts, filename):
    '''
    Save the path info as a pickle file as <filename>.pkl

        <param>
        wpts: A list of carla.Waypoint objects
        filename: Output file name
    '''
    base, ext = os.path.splitext(filename)
    count = 1
    while os.path.exists(filename):
        filename = f"{base}_{count}{ext}"
        count += 1

    wpts_list = []
    for wp in wpts:
        wp_data = {
            'location_x': wp.transform.location.x,
            'location_y': wp.transform.location.y,
            'location_z': wp.transform.location.z,
            # 可以根据需要添加其他属性
        }
        wpts_list.append(wp_data)

    with open(filename, 'wb') as f:
        pickle.dump(wpts_list, f)

def load_waypoints(filename, map):
    '''
    return the carla.Waypoint
    '''
    waypoints = []
    with open(filename, 'rb') as f:
        waypoints_data = pickle.load(f)
    for wp_data in waypoints_data:
        location = carla.Location(wp_data['location_x'], wp_data['location_y'], wp_data['location_z'])
        wp = map.get_waypoint(location)
        waypoints.append(wp)
    # print(f"Path is loaded.\nlength:{len(waypoints)*0.1}\nnum_of_points:{len(waypoints)}")
    return waypoints

def load_wp_curve(filename):
    '''
    return the coordinates of the waypoints and the coresponding curvature
    '''
    data = pd.read_csv(filename + '.csv')
    wp=data[['smoothed_x', 'smoothed_y']].values.tolist()
    curve=data['curvature'].values.tolist()

    return np.array(wp), np.array(curve)
    

def draw_waypoints(world, waypoints, z=0.1, lifetime=-1, string=False):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    i = 0
    for wpt in waypoints[1:]:
        
        begin = carla.Location(x=wpt[0], y=wpt[1], z=z)
        #begin = carla.Location(x=wpt.transform.location.x, y=wpt.transform.location.y, z=z)
        # world.debug.draw_arrow(begin, end, arrow_size=0.1, life_time=lifetime)
        if i % 10 == 0:
            world.debug.draw_point(begin, size=0.1, color=carla.Color(255,0,0), life_time=lifetime)
        
        mark = str(i)
        if string:
            world.debug.draw_string(begin, mark, color=carla.Color(0,0,255), life_time=lifetime, persistent_lines=True)
        i += 1


def find_lookahead_waypoint(location, cur_idx, lkd, wp_list):
    '''
    Find out the corresponding waypoint respact to given lookahead distance.

        <param>
        cur_idx: Current waypoint index in referance path
        lkd: Look-ahead distance (unit: meter)
        wp_list: A list of carla.Waypoint objects according to the referance path.
    输入当前的路点idx和前视距离,返回参考点中离前视距离最近的路点idx
    '''
    n = len(wp_list)
    cur = location
    for i in range(cur_idx + 1, n):
        tar = wp_list[i]
        dis = math.sqrt((tar[0] - cur[0])**2 + (tar[1] - cur[1])**2)
        if dis > lkd:
            return wp_list[i]
    
    return wp_list[cur_idx + 1]
    

def show_all_waypoints(world, distance:float):
    '''
    画出间隔为distance的所有导航点
    '''
    all_waypoints = world.get_map().generate_waypoints(distance)
    for waypoint in all_waypoints:
        world.debug.draw_point(waypoint.transform.location + carla.Location(0,0,1), size=0.1, color=carla.Color(0,255,0), life_time=30)


def show_all_spawn_points(world):
    '''
    画出所有重生点, 从0开始标号
    '''
    i = 0
    all_spawn_points = world.get_map().get_spawn_points()
    for sp in all_spawn_points:
        world.debug.draw_point(sp.location + carla.Location(0,0,3), size=0.1, color=carla.Color(255,0,0), life_time=30)
        idx = str(i)
        world.debug.draw_string(sp.location, idx, draw_shadow=False,
                      color=carla.Color(0,0,255),
                      life_time=30,
                      persistent_lines=True)
        
# [version3]
def curvature_yaw_diff(route):
    yaw = [math.radians(p.transform.rotation.yaw) for p in route]
    x = [p.transform.location.x for p in route]
    y = [p.transform.location.y for p in route]
    dists = np.array([np.hypot(dx, dy) for dx, dy in zip(np.diff(x), np.diff(y))])
    d_yaw = np.diff(make_angles_continuous(yaw))
    curvatures = d_yaw / dists
    curvatures = np.nan_to_num(curvatures, nan=0.0)
    curvatures = np.concatenate([curvatures, [0.0]])
    curvatures = [0 if abs(val)<0.001 else val for val in curvatures]
    for i in range(1,len(curvatures)-1):
        if curvatures[i-1] == 0 and curvatures[i+1]==0:
            curvatures[i] = 0
    return curvatures

def make_angles_continuous(angles):
    angles = np.array(angles)
    for i in range(len(angles)-1):
        d_angle = angles[i+1] - angles[i]
        if d_angle >= np.pi:
            angles[i+1:] -= 2.0 * np.pi
        elif d_angle <= -np.pi:
            angles[i+1:] += 2.0 * np.pi
    return angles

# [version6]
def draw_acceleration_vector(world, vehicle, accel, lifetime=1):
    begin = vehicle.get_location()
    end = begin + carla.Location(x=accel.x, y=accel.y)
    world.debug.draw_arrow(begin, end, arrow_size=0.1, life_time=lifetime)