from tile import IHT, tiles
from math import floor

"""
二维曲面监督学习，(x,y,z)
x,y 是坐标，z代表高度
x,y为 [1,3]
因为库tile使用10x10 grid tilings, 所以需要将点放大到10x10的区域，创建自己的接口

"""

maxSize = 2048
iht = IHT(maxSize)
weights = [0] * maxSize
numTilings = 8
stepSize = 0.1 / numTilings


def mytiles(x, y):
    scaleFactor = 10.0 / (3 - 1)
    return tiles(iht, numTilings, [x * scaleFactor, y * scaleFactor])


def learn(x, y, z):
    tiles = mytiles(x, y)
    estimate = 0
    for tile in tiles:
        estimate += weights[tile]
    error = z - estimate
    for tile in tiles:
        weights[tile] += stepSize * error


def test(x, y):
    tiles = mytiles(x, y)
    estimate = 0
    for tile in tiles:
        estimate += weights[tile]
    return estimate


def rectangular_grid_tilings(x, y):
    floats = [x * 4, y * 4]
    print([floor(f * 8) for f in floats])
    return tiles(iht, 8, [x * 4, y * 4])


if __name__ == "__main__":
    cc = rectangular_grid_tilings(0.7, 0.3)
    cc2 = rectangular_grid_tilings(0.7, 0.4)
    print(cc)
    print(cc2)
