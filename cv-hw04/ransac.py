from typing import List, Union
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils


TRAIN_IMG = os.path.join(os.getcwd(), 'data', 'foto1A.jpg')
QUERY_IMG = os.path.join(os.getcwd(), 'data', 'foto1B.jpg')


def main():
    trainImg = cv2.cvtColor(cv2.imread(TRAIN_IMG), cv2.COLOR_BGR2RGB)
    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
    queryImg = cv2.cvtColor(cv2.imread(QUERY_IMG), cv2.COLOR_BGR2RGB)
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)

    kpsA, featuresA = utils.detectAndDescribe(trainImg_gray, 'orb')
    kpsB, featuresB = utils.detectAndDescribe(queryImg_gray, 'orb')

    matches = utils.matchKeyPointsBF(featuresA, featuresB, method='orb')
    _, M, status = utils.getHomography(kpsA, kpsB, featuresA, featuresB,
                                       matches, 4)
    print(M)
    print(status.sum())
    # H = findHomography(kpsA, kpsB, matches, 5, 5.0, 100, True)
    H = findHomography_v2(kpsA, kpsB, matches, 8, 2.0, None, 0.9999, True)
    print(H)

    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]
    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
    plt.figure(figsize=(20, 10))
    plt.imshow(result)
    plt.axis('off')
    plt.show()


def findHomography(kpsA: List[cv2.KeyPoint],
                   kpsB: List[cv2.KeyPoint],
                   matches: List[cv2.DMatch],
                   s: int = 4,
                   reprojThresh: float = 4.0,
                   num_iterations: int = 100,
                   use_all: bool = False):
    if len(matches) <= s:
        return None

    # find coordinates of matched points
    ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

    return ransac(ptsA, ptsB, s, reprojThresh, num_iterations, use_all)


def ransac(ptsA: np.ndarray, ptsB: np.ndarray, s: int = 4,
           reprojThresh: float = 4.0, num_iterations: int = 100,
           use_all: bool = False):
    n = len(ptsA)
    assert len(ptsB) == n

    # initialize
    H_best = np.zeros((3, 3), dtype='float32')
    outliers_best = len(ptsA)
    inliers = np.arange(0)

    # outer loop: try num_iterations random permutations
    for _ in range(num_iterations):
        inds = np.random.permutation(range(n))
        i1, i2 = 0, s
        while i2 <= n:
            # find homography matrix by least squares
            inds_in = inds[i1:i2]
            H = estimate_homography(ptsA[inds_in], ptsB[inds_in])

            # check the number of outliers
            inds_out = np.concatenate([inds[:i1], inds[i2:]])
            pred = transform(ptsA[inds_out], H)
            distances = np.sqrt(np.sum((ptsB[inds_out] - pred)**2, axis=1))
            mask = (distances <= reprojThresh)
            outliers_cur = (~mask).sum()
            if outliers_cur < outliers_best:
                H_best = H
                outliers_best = outliers_cur
                inliers = np.concatenate([inds_in, inds_out[mask]])

            # update indices
            i1 = i2
            i2 = i1 + s

    # recalculate homography matrix using all inliers
    print(f'number of inliers: {n - outliers_best}')
    if use_all:
        H_best = estimate_homography(ptsA[inliers], ptsB[inliers])
    return H_best


def findHomography_v2(kpsA: List[cv2.KeyPoint],
                      kpsB: List[cv2.KeyPoint],
                      matches: List[cv2.DMatch],
                      s: int = 4,
                      reprojThresh: float = 4.0,
                      num_iterations: Union[int, None] = None,
                      p: float = 0.99,
                      use_all: bool = False):
    if len(matches) <= s:
        return None

    # find coordinates of matched points
    ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

    return ransac_v2(ptsA, ptsB, s, reprojThresh, num_iterations, p, use_all)


def ransac_v2(ptsA, ptsB, s=4, reprojThresh=4.0, num_iterations=None,
              p=0.99, use_all=False):
    n = len(ptsA)
    assert len(ptsB) == n

    # initialization
    H_best = np.zeros((3, 3), dtype='float32')
    outliers_best = len(ptsA)
    inliers = np.arange(0)

    # use adaptive strategy if number of iterations not specified
    if num_iterations is None:
        num_iterations = np.inf
        adaptive = True
    else:
        adaptive = False
    sample_count = 0

    # perform iterations
    while sample_count < num_iterations:
        # split points into 2 subsets
        inds = np.random.permutation(n)
        inds_in = inds[:s]
        inds_out = inds[s:]
        sample_count += 1

        # find homography matrix by least squares
        H = estimate_homography(ptsA[inds_in], ptsB[inds_in])

        # check the number of outliers
        pred = transform(ptsA[inds_out], H)
        distances = np.sqrt(np.sum((ptsB[inds_out] - pred)**2, axis=1))
        mask = (distances <= reprojThresh)
        outliers_cur = (~mask).sum()
        if outliers_cur < outliers_best:
            H_best = H
            outliers_best = outliers_cur
            inliers = np.concatenate([inds_in, inds_out[mask]])
            if adaptive:  # update number of iterations
                e = outliers_best / n
                num_iterations = np.log(1 - p) / np.log(1 - (1 - e)**s)

    # recalculate homography matrix using all inliers
    print(f'number of inliers: {n - outliers_best}')
    if use_all:
        H_best = estimate_homography(ptsA[inliers], ptsB[inliers])
    return H_best


def estimate_homography(src, dst):
    A = []
    for (x1, y1), (x2, y2) in zip(src, dst):
        A.append([-x1, -y1, -1, 0, 0, 0, x2 * x1, x2 * y1, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1 * y2, y1 * y2, y2])
    A = np.array(A)
    _, _, v = np.linalg.svd(A)
    H = v[-1].reshape(3, 3).astype('float32')
    H /= H[2, 2]
    return H


def transform(src, H):
    inp = np.hstack([src, np.ones((len(src), 1))])
    out = H @ inp.T
    pred = np.zeros((len(src), 2), dtype='float32')
    pred[:, 0] = out[0] / out[2]
    pred[:, 1] = out[1] / out[2]
    return pred


if __name__ == '__main__':
    main()
