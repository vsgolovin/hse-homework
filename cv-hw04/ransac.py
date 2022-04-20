from typing import List
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
    H = findHomography(kpsA, kpsB, matches, 4, 5.0)
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
                   reprojThresh: float = 4.0):
    if len(matches) <= s:
        return None

    # find coordinates of matched points
    ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

    return ransac(ptsA, ptsB, s, reprojThresh)


def ransac(ptsA: np.ndarray, ptsB: np.ndarray, s: int, reprojThresh: float,
           num_iterations: int = 100):
    n = len(ptsA)
    assert len(ptsB) == n

    H_best = np.empty((3, 3), dtype='float32')
    H_best[-1, :] = [0, 0, 1]
    outliers_best = len(ptsA)
    inliers = np.arange(0)

    for _ in range(num_iterations):
        inds = np.random.permutation(range(n))
        i1, i2 = 0, s

        while i2 <= n:
            inds_in = inds[i1:i2]
            src = np.hstack([ptsA[inds_in], np.ones((s, 1))])
            dst = ptsB[inds_in]
            H = np.linalg.lstsq(src, dst, rcond=None)[0]

            inds_out = np.concatenate([inds[:i1], inds[i2:]])
            src = np.hstack([ptsA[inds_out], np.ones((n - s, 1))])
            dst = ptsB[inds_out]
            distances = np.sqrt(np.sum((dst - src @ H)**2, axis=1))
            mask = (distances <= reprojThresh)
            outliers_cur = (~mask).sum()
            if outliers_cur < outliers_best:
                H_best[0:2, :] = H.T
                outliers_best = outliers_cur
                inliers = np.concatenate([inds_in, inds_out[mask]])
            i1 = i2
            i2 = i1 + s

    src = np.hstack([ptsA[inliers], np.ones((len(inliers), 1))])
    H_best[0:2, :] = np.linalg.lstsq(src, ptsB[inliers], rcond=None)[0].T
    return H_best


if __name__ == '__main__':
    main()
