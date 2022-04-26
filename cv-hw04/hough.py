import os
import copy
import heapq
import numpy as np
import matplotlib.pyplot as plt
import cv2


IMG_PATH = os.path.join(os.getcwd(), 'data', 'coins_1.jpg')


def main():
    img = cv2.imread(IMG_PATH)
    blur = cv2.GaussianBlur(img, (7, 7), 1)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 180, 220)

    # use implemented algorithm
    A, radii = hough_transform(edges, 20, 60, 20)
    circles = filter_hough_circles(A, radii, 55, 50)
    print(f'Found {len(circles)} circles.')
    img2 = copy.deepcopy(img)
    for x, y, r in circles:
        cv2.circle(img2, (x, y), r, color=(0, 255, 0), thickness=2)
    plt.figure()
    plt.imshow(img2)

    # use cv2.HoughCircles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=40,
        minRadius=20,
        maxRadius=60,
        param2=50
    )
    img3 = copy.deepcopy(img)
    for circle in circles.squeeze():
        x0, y0, r = np.uint16(circle)
        cv2.circle(img3, (x0, y0), r, (0, 255, 0), thickness=2)
    plt.figure()
    plt.imshow(img3)

    plt.show()


def hough_circles(img, threshold,
                  canny_thresh_1=100, canny_thresh_2=200,
                  r_min=5, r_max=None, r_bins=30, min_dist=20):
    img = cv2.Canny(img, canny_thresh_1, canny_thresh_2)
    A, radii = hough_transform(img, r_min, r_max, r_bins)
    return filter_hough_circles(A, radii, threshold, min_dist)


def hough_transform(img, r_min=5, r_max=None, r_bins=30):
    h, w = img.shape
    if r_max is None:
        r_max = min(h, w) // 2
    radii = np.linspace(r_min, r_max, r_bins + 1)
    x0, y0 = np.meshgrid(np.arange(0, w), np.arange(0, h))
    A = np.zeros((h, w, r_bins), dtype='uint32')

    # iterate over pixels
    for i in range(h):
        print(f'Hough transform progress: {i}/{h - 1}', end='\r')
        for j in range(w):
            if img[i, j] == 0:
                continue
            r = np.sqrt((i - y0)**2 + (j - x0)**2)
            for k in range(r_bins):
                mask = (r >= radii[k]) & (r < radii[k + 1])
                A[:, :, k][mask] += img[i, j]
    print()

    # scale accumulator matrix so that values are <= 100
    circ = np.pi * (radii[:-1] + radii[1:])
    A = (A / (circ * 2.55))
    return A, radii


def filter_hough_circles(A, radii, threshold, min_dist=20):
    # sort circles by "intensity"
    heap = []
    ix, iy, ir = np.where(A >= threshold)
    for i, j, k in zip(ix, iy, ir):
        heapq.heappush(heap, (-A[i, j, k], (j, i, radii[k])))

    # remove circles that are too close to ones with higher intensity
    circles = []
    while heap:
        _, (x0, y0, r) = heapq.heappop(heap)
        distances = [np.sqrt((xi - x0)**2 + (yi - y0)**2)
                     for xi, yi, _ in circles]
        if not distances or min(distances) > min_dist:
            circles.append(np.array([x0, y0, r], dtype='uint16'))

    return circles


if __name__ == '__main__':
    main()
