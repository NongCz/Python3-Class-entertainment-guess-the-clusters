"""
----- Without using numpy -----
"""
import random as r
import matplotlib.pyplot as plt
import numpy as np
import math
import time

start_time = time.time()
COLORS = ['green', 'blue', 'black', 'purple']
POINTS_AT_CENTROIDS = [[] for _ in range(4)]
rand_clst_points = np.loadtxt('points.csv', delimiter=',')

x_max = np.max(rand_clst_points[:, 0])
x_min = np.min(rand_clst_points[:, 0])
y_max = np.max(rand_clst_points[:, 1])
y_min = np.min(rand_clst_points[:, 1])

guess_centroids = np.random.randint([x_min, y_min], [x_max, y_max], size=(4, 2))
new_centroids = guess_centroids


def get_distance(dist_x, dist_y):
    dist = math.sqrt(dist_x**2 + dist_y**2)
    return dist

def k_mean_algo(n):
    def assign_points():
        global at_cent 
        global POINTS_AT_CENTROIDS
        POINTS_AT_CENTROIDS = [[] for _ in range(4)]
        for point in rand_clst_points:
            at_cent = 0
            min_dist = x_max
            for idx, val in enumerate(new_centroids):
                dist = get_distance(
                    new_centroids[idx][0] - point[0],
                    new_centroids[idx][1] - point[1]
                )
                if dist < min_dist:
                    min_dist = dist
                    at_cent = idx
            POINTS_AT_CENTROIDS[at_cent].append(point) # [[[], []], []]

    def calculate_new_centroids():
        global new_centroids
        new_centroids = []
        for point_set in POINTS_AT_CENTROIDS:
            l = len(point_set)
            sum_x = 0
            sum_y = 0
            for points in point_set:
                sum_x += points[0]
                sum_y += points[1]
            mean_x = sum_x / l
            mean_y = sum_y / l
            new_centroids.append([mean_x, mean_y])

    for _ in range(n):
        assign_points()
        calculate_new_centroids()

def plot_centroids():
    c = 0
    for idx, cluster in enumerate(POINTS_AT_CENTROIDS):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], color=COLORS[idx])

    for i in range(4):
        x1 = guess_centroids[i][0]
        y1 = guess_centroids[i][1]
        x2 = new_centroids[i][0]
        y2 = new_centroids[i][1]
        plt.scatter(x1, y1, color='red', marker='+', label='Old centroid', s=100)
        plt.scatter(x2, y2, color='orange', marker='*', label='New centroid', s=100)
        print(f'Old centroids #{i}: ({x1}, {y1})')
        print(f'New centroids #{i}: ({x2}, {y2})')

k_mean_algo(500)
plot_centroids()

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Centroids and random points.')
plt.grid(True)
plt.legend()
end_time = time.time()
print(f'Program time: {end_time - start_time}')
plt.show()
