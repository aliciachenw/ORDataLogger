import numpy as np
import copy
mat = np.array([[0.060908, 0.00229319, 0.0122661, -23.386],
            [-0.00155518, -0.0568636, -0.00279538, -108.578],
            [0.0128902, 0.00320918, -0.0582963, 13.0809],
            [0, 0, 0, 1]])

# svd
U, s, V = np.linalg.svd(mat[:3, :3])
print(s)

calib_w = 698
calib_h = 727
curr_w = 227
curr_h = 390

# new_sx = calib_w * s[0] / curr_w
# new_sy = calib_h * s[1] / curr_h

new_sx = calib_w * s[0] / curr_w
new_sy = 65 / curr_h
new_sx = new_sy

new_s = np.array([new_sx, new_sy, (new_sx + new_sy) / 2])
# get the rotation matrix
rot = U @ np.diag(new_s) @ V

new_mat = copy.deepcopy(mat)
new_mat[:3, :3] = rot

for i in range(4):
    for j in range(4):
        print(new_mat[i, j], end=' ')
# print(new_mat)