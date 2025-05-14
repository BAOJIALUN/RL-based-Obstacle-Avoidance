import carla
from waypoints_utils import save_waypoints, draw_waypoints, load_waypoints
import random
import time
from global_planner import compute_route_waypoints
from env import CarEnv
actor_list = []
route_waypoints = list()

client = carla.Client("localhost", 2000)
client.set_timeout(3)
world = client.get_world()
blueprint = world.get_blueprint_library()
wmap = world.get_map()
spawn_points = wmap.get_spawn_points()

origin = spawn_points[72]
destination = spawn_points[85]
rwp = compute_route_waypoints(wmap, origin, destination, resolution=0.2)
for each in rwp:
    route_waypoints.append(each[0])
draw_waypoints(world, route_waypoints, lifetime=50)
save_waypoints(route_waypoints, "town5_waypoints_long_r0d2.pkl")
# vehicle_bp = blueprint.filter("model3")[0]
# for _ in range(1):
#     spawn_point = spawn_points[88]
#     print(spawn_point)
#     vehicle = world.spawn_actor(vehicle_bp, spawn_point)
#     time.sleep(2)


#     number_of_waypoints = 2000
#     waypoint = map.get_waypoint(vehicle.get_location(), project_to_road=True, lane_type=(carla.LaneType.Driving))
#     current_waypoint = waypoint
#     route_waypoints.append(current_waypoint)
#     for x in range(number_of_waypoints):
#         next_waypoint = current_waypoint.next(0.2)[0]
#         route_waypoints.append(next_waypoint)
#         current_waypoint = next_waypoint
#     save_waypoints(route_waypoints, "town3_waypoints_hdcurve.pkl")

#     if vehicle:
#         vehicle.destroy()