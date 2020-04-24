import numpy as np
import matplotlib.pyplot as plt

def get_centers_distribution(imgs):
  distribution = []

  for img in imgs:
    max_amp = np.max(img)
    min_amp = np.min(img)
    normilized_img = (img - min_amp) / (max_amp - min_amp)

    sum, x_sum, y_sum = 0, 0, 0
    for x in range(0, 8):
      for y in range(0, 8):
        x_sum += normilized_img[x][y] * x
        y_sum += normilized_img[x][y] * y
        sum += normilized_img[x][y]

    distribution.append([x_sum / sum, y_sum / sum])

  return np.array(distribution)

def centers_distribution(real, generated):
  original_distribution = get_centers_distribution(real)
  generated_distribution = get_centers_distribution(generated)
  plt.scatter(original_distribution[:, 0], original_distribution[:, 1], color='limegreen')
  plt.scatter(generated_distribution[:, 0], generated_distribution[:, 1], color='gold')
  plt.show()
  original_centers = (np.mean(original_distribution[:, 0]),np.mean(original_distribution[:, 1]))
  generated_centers = ((np.mean(generated_distribution[:, 0])), np.mean(generated_distribution[:, 1]))
  return (original_centers, generated_centers)