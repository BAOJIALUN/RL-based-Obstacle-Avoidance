import carla
from waypoints_utils import save_waypoints, draw_waypoints, load_waypoints, load_wp_curve
import random
import time
import os

actor_list = []
filename = "town3_waypoints_turn_right_smoothed"
route_waypoints = list()

client = carla.Client("localhost", 2000)
client.set_timeout(3)
if 'town3' in filename:
    client.load_world("Town03")
world = client.get_world()
blueprint = world.get_blueprint_library()
map = world.get_map()


waypoints=list()

waypoints, curvature = load_wp_curve(filename)
draw_waypoints(world, waypoints, lifetime=50)