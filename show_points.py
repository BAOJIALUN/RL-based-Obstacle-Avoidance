'''
20240411_1524已测试 
'''


import carla
import time
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("--ws", required=True, type=str, help="wp for drawing waypoints in 5m, sp for drawing all spawn points")
args = vars(ap.parse_args())

client = carla.Client('localhost', 2000)
client.set_timeout(10)

world = client.get_world()
map = world.get_map()

all_spawn_points = map.get_spawn_points()
all_waypoints = map.generate_waypoints(5)


debug = world.debug

if args['ws'] == 'sp' :
    i=0
    for spawn_point in all_spawn_points:
        debug.draw_point(spawn_point.location + carla.Location(0, 0, 5), size=0.1, color=carla.Color(255,0,0), life_time=-1)
        mark = str(i)
        debug.draw_string(spawn_point.location, mark, draw_shadow=False,
                        color=carla.Color(0,0,255),
                        life_time=600,
                        persistent_lines=True)
        i += 1
elif args['ws'] == 'wp':
    s=0
    for waypoint in all_waypoints:
        debug.draw_point(waypoint.transform.location + carla.Location(0,0,1), size=0.1, color=carla.Color(0,255,0), life_time=60)

else:
    pass