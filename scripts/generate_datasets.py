import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

# Dataset 1: Separated by a straight line
def generate_dataset1(num_points, noise_level):
    X = np.random.rand(num_points, 2) * 2 - 1
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X += np.random.normal(0, noise_level, X.shape)
    return X, y

# Dataset 2: Points labeled 1 around points labeled 0
def generate_dataset2(num_points, noise_level):
    X_zero = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], num_points // 2)
    radius = np.random.uniform(3, 5, num_points // 2)
    angle = np.random.uniform(0, 2 * np.pi, num_points // 2)
    X_one = np.array([radius * np.cos(angle), radius * np.sin(angle)]).T
    X = np.vstack((X_zero, X_one))
    y = np.concatenate((np.zeros(num_points // 2), np.ones(num_points // 2)))
    X += np.random.normal(0, noise_level, X.shape)
    return X, y

# Dataset 3: Separated by a tilted and less wavy line
def generate_dataset3(num_points, noise_level):
    X = np.random.rand(num_points, 2) * 5 - 2.5
    y = (0.5 * X[:, 0] + np.sin(4 * X[:, 0])*0.7 > X[:, 1]).astype(int)
    X += np.random.normal(0, noise_level, X.shape)
    return X, y

# Dataset 4: Two interleaved spirals
def generate_dataset4(num_points, noise_level):
    t = np.linspace(1, 5 * np.pi, num_points)
    r = t
    X_zero = np.array([r * np.cos(t), r * np.sin(t)]).T
    X_one = np.array([-r * np.cos(t), -r * np.sin(t)]).T
    X = np.vstack((X_zero, X_one))
    y = np.concatenate((np.zeros(num_points), np.ones(num_points)))
    X += np.random.normal(0, noise_level, X.shape)
    X /= 10.0
    return X, y

def generate_dataset5(num_points, noise_level):
    # Total number of points
    num_total = num_points

    # Number of points for each region
    num_red_donut = num_total // 3
    num_red_circle = num_total // 3
    num_blue_background = (num_total - num_red_donut - num_red_circle)*5

    # Generate red donut points centered at (0, 0)
    theta_donut = np.random.uniform(0, 2 * np.pi, num_red_donut)
    r_donut = np.random.uniform(2.0, 3.0, num_red_donut)
    X_donut = np.column_stack((
        r_donut * np.cos(theta_donut),
        r_donut * np.sin(theta_donut)
    ))
    y_donut = np.ones(num_red_donut)

    # Generate red circle points centered at (7, 3)
    theta_circle = np.random.uniform(0, 2 * np.pi, num_red_circle)
    r_circle = np.random.uniform(0.0, 1.0, num_red_circle)
    X_circle = np.column_stack((
        7 + r_circle * np.cos(theta_circle),
        3 + r_circle * np.sin(theta_circle)
    ))
    y_circle = np.ones(num_red_circle)

    # Generate blue background points, excluding red regions
    blue_points = []
    while len(blue_points) < num_blue_background:
        # Generate candidate points
        num_candidates = num_blue_background * 2  # Generate more candidates than needed
        X_candidate = np.random.uniform(-5, 12, (num_candidates, 2))

        # Calculate distances from red regions
        r_from_origin = np.sqrt(X_candidate[:, 0]**2 + X_candidate[:, 1]**2)
        r_from_circle_center = np.sqrt((X_candidate[:, 0] - 7)**2 + (X_candidate[:, 1] - 3)**2)

        # Conditions to be outside the red donut and red circle
        outside_donut = (r_from_origin < 2.0) | (r_from_origin > 3.0)
        outside_circle = r_from_circle_center > 1.0

        # Keep points that are outside both red regions
        valid_indices = np.where(outside_donut & outside_circle)[0]
        X_valid = X_candidate[valid_indices]

        # Add valid points to the blue points list
        blue_points.extend(X_valid.tolist())

    # Trim to the required number of blue points
    X_background = np.array(blue_points[:num_blue_background])
    y_background = np.zeros(num_blue_background)

    # Combine all points and labels
    X = np.vstack((X_donut, X_circle, X_background))
    y = np.concatenate((y_donut, y_circle, y_background))

    # Add Gaussian noise
    X += np.random.normal(0, noise_level, X.shape)

    return X, y

# Generate datasets
num_points = 1000
noise_level = 0.1

X1, y1 = generate_dataset1(num_points, noise_level)
X2, y2 = generate_dataset2(num_points, noise_level)
X3, y3 = generate_dataset3(num_points, noise_level)
X4, y4 = generate_dataset4(num_points, noise_level*3)
X5, y5 = generate_dataset5(num_points, noise_level)

# Export datasets as CSV files
pd.DataFrame(np.hstack((X1, y1.reshape(-1, 1))), columns=['X1', 'X2', 'y']).to_csv('flat.csv', index=False)
pd.DataFrame(np.hstack((X2, y2.reshape(-1, 1))), columns=['X1', 'X2', 'y']).to_csv('donut.csv', index=False)
pd.DataFrame(np.hstack((X3, y3.reshape(-1, 1))), columns=['X1', 'X2', 'y']).to_csv('wavy.csv', index=False)
pd.DataFrame(np.hstack((X4, y4.reshape(-1, 1))), columns=['X1', 'X2', 'y']).to_csv('spiral.csv', index=False)
pd.DataFrame(np.hstack((X5, y5.reshape(-1, 1))), columns=['X1', 'X2', 'y']).to_csv('donut_circle.csv', index=False)

# Render graphs of datasets
#fig, axs = plt.subplots(1, 4, figsize=(20, 5))

#for i, (X, y) in enumerate([(X1, y1), (X2, y2), (X3, y3), (X4, y4)]):
#    axs[i].scatter(X[y == 0, 0], X[y == 0, 1], label='0', alpha=0.7)
#    axs[i].scatter(X[y == 1, 0], X[y == 1, 1], label='1', alpha=0.7)
#    axs[i].set_title(f'Dataset {i+1}')
#    axs[i].legend()

#plt.tight_layout()
#plt.show()