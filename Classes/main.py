import glob

from CV_Camera_Distance import take_mapping_image, reconstruction_multi
from CV_Camera_Distance import mapping, route_plan
from Pi_Car_Class import Pi_Car
from Object_Detection import goal


def main():
    # Initialization conditions to generate a new map
    car = Pi_Car()
    take_mapping_image()
    car.forward()
    take_mapping_image()
    images = glob.glob("Mapping Images/*.jpg")

    start_point, points = reconstruction_multi(images)
    start_point, grid_map = mapping(start_point, points, images[-1])
    goal_point = goal()
    movements = route_plan(start_point, goal_point, grid_map)

    # Follow path
    current_movement = ""
    for movement in movements:
        if movement == "S":
            car.forward()
            current_movement = "S"
        elif movement == "N":
            car.reverse()
            current_movement = "N"
        elif movement == "E":
            if current_movement != "E":
                car.rotate_right()
                car.forward()
            else:
                car.forward()
            current_movement = "E"
        elif movement == "W":
            if current_movement != "W":
                car.rotate_left()
                car.forward()
            else:
                car.forward()
            current_movement = "W"

        take_mapping_image()
        images = glob.glob("Mapping Images/*.jpg")[-2:]
        start_point, points = reconstruction_multi(images)
        start_point, grid_map = mapping(start_point, points, images[-1])
        goal_point = goal()
        movements = route_plan(start_point, goal_point, grid_map)


if __name__ == "main":
    main()
