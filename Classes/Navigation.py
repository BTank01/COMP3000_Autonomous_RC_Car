import numpy as np
import random
import matplotlib.pyplot as plt


# RTT Algorithm
def is_valid_point(point, map, shape):
    x, y = point
    x_valid = 0 <= x < shape[0]
    y_valid = 0 <= y < shape[1]  
    if (x > 0 and x < shape[0]):
        if (y >= 0 and y < shape[1]):
            return map[int(x), int(y)] == 0
    return False


def find_random_point(shape):
    return (random.uniform(0, shape[0]), random.uniform(0, shape[1]))


def distance_metric(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def nearest_node(tree, point):
    min_dist = float('inf')
    nearest = None
    for node in tree:
        dist = distance_metric(node, point)
        if dist < min_dist:
            min_dist = dist
            nearest = node
    return nearest


# Function to extend tree towards a point
def extend_tree(tree, point, step_size, map, shape):
    nearest = nearest_node(tree, point)
    direction = (point[0] - nearest[0], point[1] - nearest[1])
    distance = distance_metric(nearest, point)
    if distance > step_size:
        direction = (direction[0] / distance * step_size, direction[1] / distance * step_size)
        new_point = (nearest[0] + direction[0], nearest[1] + direction[1])
    else:
        new_point = point
    if is_valid_point(new_point, map, shape):
        return new_point
    else:
        return None


# Function to find path using RRT
def rrt(map, shape, start, goal, max_iterations=1000, step_size=0.1):
    tree = [start]  # Initialize tree with start node
    for _ in range(max_iterations):
        random_point = find_random_point(shape)
        new_point = extend_tree(tree, random_point, step_size, map, shape)
        if new_point is not None:
            if np.allclose(new_point, goal):
                # Reached goal, return path
                path = [goal]
                current = tree[np.argmin([distance_metric(n, goal) for n in tree])]
                while not np.allclose(current, start):
                    path.append(current)
                    current = nearest_node(tree, current)
                path.append(start)
                return path[::-1]
            else:
                tree.append(new_point)
    return None  # No path found


# A* Algorithm
def astar(grid_map, start, goal):
    # Set Movements
    movements = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    movement_names = ['S', 'E', 'N', 'W']

    # Initialize starting conditions
    closed = np.zeros_like(grid_map)
    closed[start] = 1

    cost = {start: 0}
    parent = {start: None}
    movements_taken = {}

    # Initialize priority queue
    pq = [(0, start)]

    # A* algorithm loop
    while pq:
        # Get node with lowest cost from priority queue
        current_cost, current_node = pq.pop(0)

        # Check if goal reached
        if current_node == goal:
            path = []
            movements = []
            while current_node is not None:
                path.append(current_node)
                if current_node in movements_taken:
                    movements.append(movement_names[movements_taken[current_node]])
                current_node = parent[current_node]
            return path[::-1], movements[::-1]

        # Explore neighbors
        for i, (dx, dy) in enumerate(movements):
            neighbor = (current_node[0] + dx, current_node[1] + dy)
            # Check if neighbor is within bounds and not an obstacle
            if 0 <= neighbor[0] < grid_map.shape[0] and 0 <= neighbor[1] < grid_map.shape[1] and grid_map[neighbor] == 0:
                # Calculate cost to reach neighbor
                new_cost = cost[current_node] + np.linalg.norm(np.array(neighbor) - np.array(current_node))
                # Update cost, parent, and movements if new path is cheaper
                if neighbor not in cost or new_cost < cost[neighbor]:
                    cost[neighbor] = new_cost
                    priority = new_cost + np.linalg.norm(np.array(neighbor) - np.array(goal))
                    pq.append((priority, neighbor))
                    parent[neighbor] = current_node
                    # Determine movement direction for the neighbor
                    dx_parent = neighbor[0] - current_node[0]
                    dy_parent = neighbor[1] - current_node[1]
                    movements_taken[neighbor] = movements.index((dx_parent, dy_parent))
                    closed[neighbor] = 1
    # No path found
    return None, None


def test():
    grid_map = np.array([[0, 0, 0, 0, 0],
                        [0, 1, 1, 1, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0]])

    start = (0, 0)
    goal = (4, 4)

    path, movements = astar(grid_map, start, goal)
    print("Path:", path)
    print("Movements:", movements)


# test()
