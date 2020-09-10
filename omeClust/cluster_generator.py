#!/usr/bin/env python

import sys
import numpy as np
import math


def nearest_spd(M):
    # Find the nearest SPD matrix to M
    # http://www.sciencedirect.com/science/article/pii/0024379588902236
    U, s, V = np.linalg.svd(M)
    H = V * s * V.T
    Ms = (M + H) / 2
    return (Ms + Ms.T) / 2


def hard_cov_dataset_generate(nSamples, C, nX, nY):
    # Find a nearby SPD matrix to C, maintaining the zero/nonzero structure of C
    Z = abs(C) < 10 * sys.float_info.epsilon
    Sc = math.sqrt(np.sum(pow(C, 2)))
    for i in range(0, 500):
        E, V = np.linalg.eig(C)
        if not any(E < 100 * sys.float_info.epsilon):
            break

        C = nearest_spd(C)
        C[Z] = 0.0
        C = C * Sc / math.sqrt(np.sum(pow(C, 2)))

    # Ensure there is no lingering floating point silliness
    E, V = np.linalg.eig(C)
    while any(E < 100 * sys.float_info.epsilon):
        C = C * 0.999 + 0.001 * np.eye(nX + nY)
        E, V = np.linalg.eig(C)

    # Generate a dataset
    xy = np.random.multivariate_normal(np.zeros(nX + nY), C, nSamples)

    # Split it into pieces to get the two paired datasets
    x = xy[:, :nX].T
    y = xy[:, nX:].T

    return (x, y, C[:nX, nX:])


def circular_continuous(nSamples, nX, nY, noiseVar, autoCov, lengthScaleX, lengthScaleY, crossCov, crossLengthScale,
                        fractionZero):
    # Make masks for the parts of the matrix representing X and Y
    isX = np.concatenate([np.ones(nX), np.zeros(nY)])
    isY = np.concatenate([np.zeros(nX), np.ones(nY)])
    pairX = np.multiply.outer(isX, isX)
    pairY = np.multiply.outer(isY, isY)

    # Within-dataset "time" dimension
    t = np.concatenate([np.linspace(0, 2 * math.pi, num=nX, endpoint=False),
                        np.linspace(0, 2 * math.pi, num=nY, endpoint=False)])
    dt = np.subtract.outer(t, t)
    periodicCov = (pairX * autoCov * np.exp(-np.abs(np.sin(dt)) / lengthScaleX) +
                   pairY * autoCov * np.exp(-np.abs(np.sin(dt)) / lengthScaleY) +
                   np.maximum(0.0, (1 - pairX - pairY) * crossCov * np.exp(
                       -np.abs(np.sin(dt)) / crossLengthScale) - fractionZero * crossCov))

    # Put it all together
    C = noiseVar * np.eye(nX + nY) + periodicCov

    return hard_cov_dataset_generate(nSamples, C, nX, nY)


def circular_block(nSamples, nX, nY, nBlocks, noiseVar, blockIntraCov, offByOneIntraCov, blockInterCov,
                   offByOneInterCov, holeCov, holeProb):
    # Make masks for the parts of the matrix representing X and Y
    isX = np.concatenate([np.ones(nX), np.zeros(nY)])
    isY = np.concatenate([np.zeros(nX), np.ones(nY)])
    pairX = np.multiply.outer(isX, isX)
    pairY = np.multiply.outer(isY, isY)

    # Generate some random clusters
    clusterID = list(np.sort([np.random.randint(1, nBlocks) for i in range(0, nX)])) + \
                list(np.sort([np.random.randint(1, nBlocks) for i in range(0, nY)]))
    # print clusterID
    # Get the differences between clusters
    dClust = np.minimum(
        np.abs(np.subtract.outer(clusterID, clusterID)),
        np.abs(np.subtract.outer(clusterID, np.array(clusterID) - nBlocks)))

    # Generate intra-dataset and inter-dataset covariance matrices
    intraC = (blockIntraCov * (dClust == 0) + offByOneIntraCov * (dClust == 1)) * (pairX + pairY)
    interC = (blockInterCov * (dClust == 0) + offByOneInterCov * (dClust == 1)) * (1 - pairX - pairY)

    # Put it all together
    C = noiseVar * np.eye(nX + nY) + intraC + interC

    # introduce holes in C
    holes = np.Infinity * np.ones((nX + nY, nX + nY))
    holePos = np.random.uniform(low=0, high=1, size=(nX + nY, nX + nY)) < holeProb
    holePos[np.tril_indices(nX + nY, 1)] = False
    holes[holePos] = holeCov
    holes = np.minimum(holes, holes.T)
    C = np.minimum(C, holes)
    # print C
    return hard_cov_dataset_generate(nSamples, C, nX, nY)


def rope_unrelated(nSamples, nX, nY, noiseVar, ropeMinCov, ropeMaxCov, fractionTruePositive, crossDatasetCov):
    # Make masks for the parts of the matrix representing X and Y
    isX = np.concatenate([np.ones(nX), np.zeros(nY)])
    isY = np.concatenate([np.zeros(nX), np.ones(nY)])
    pairX = np.multiply.outer(isX, isX)
    pairY = np.multiply.outer(isY, isY)

    # Build the within-dataset ropes
    cI = np.concatenate([np.linspace(ropeMinCov, ropeMaxCov, num=nX),
                         np.linspace(ropeMinCov, ropeMaxCov, num=nY)])
    ropeCov = (pairX + pairY) * np.minimum.outer(cI, cI)

    # Cross-dataset correlations
    xyCov = crossDatasetCov * (np.random.rand(nX, nY) < fractionTruePositive)
    crossCov = np.vstack([np.hstack([np.zeros((nX, nX)), xyCov]), np.hstack([xyCov.T, np.zeros((nY, nY))])])

    # Put it all together
    C = noiseVar * np.eye(nX + nY) + ropeCov + crossCov
    return hard_cov_dataset_generate(nSamples, C, nX, nY)

# circular_continuous(5, 7, 11, 0.1, 0.5, 0.3, 0.3, 0.3, 0.25, 0.01)
# rope_unrelated(5, 3, 4, 0.1, 0.4, 0.7, 0.5, 0.3)

# A strong relationship exists when the covariance (`crossCov` in `circular_continuous` or `crossDatasetCov` in `rope_unrelated`)
# is near the variance of the variables themselves, which is `noiseVar` + (`autoCov` or `ropeMaxCov`)
# now in the first one, `fractionZero` should be 0-1
# it won't be exactly that many zeros, but bigger numbers will make more zeros
# the other one, you tweak `fractionTruePositive`

# X,Y,A = omeClust.cluster_generator.circular_block(nSamples = 100, nX =100, nY = 100, nBlocks =5, noiseVar = 0.1,
#                                             blockIntraCov = 0.3, offByOneIntraCov = 0.0,
#                                             blockInterCov = 0.2, offByOneInterCov = 0.0,
#                                             holeCov = 0.3, holeProb = .25)
