import numpy as np
import time

def get_distance2_matrix(centers, moving=None):
    dists = np.zeros((centers.shape[0], centers.shape[0], 2))
    dists[:] = centers
    if not (moving is None):
        d = dists[moving, :, :]
        c = centers[moving, :]
    else:
        d = dists
        c = centers

    d[:, :, 0] -= c[:, 0][:, None]
    d[:, :, 1] -= c[:, 1][:, None]

    return d[:, :, 0]*d[:, :, 0] + d[:, :, 1]*d[:,:,1]

def get_collisions(centers, r2=1, moving=None):
    dists = get_distance2_matrix(centers, moving)
    i1, i2 = np.where(dists < r2)
    i1_ = (np.arange(centers.shape[0])[moving])[i1]
    same = i2 != i1_
    return i2[same], i1_[same]

def get_average_distance(centers):
    closest_circs = np.min(np.sqrt(get_distance2_matrix(centers)) + np.identity(centers.shape[0]) * 1e10, axis=1)
    return np.mean(closest_circs)

def unify_speed(v, uni_speed):
    v_norm = np.sqrt(np.sum(v**2, axis=1))
    return v/v_norm[:,None] * uni_speed

def correct_center(v1, v2, c1, c2, r2):
    delta_v = v1 - v2
    alpha = c1 - c2
    cur_dist = (alpha**2).sum()
    delta_v_norm = (delta_v**2).sum()
    scalar = np.multiply(alpha,delta_v).sum()
    lambda_correct = (scalar + 0.5 * np.sqrt(4*scalar**2 - 4*delta_v_norm*(cur_dist-r2)))/delta_v_norm
    return lambda_correct

def update_vectors(x1_inds, x2_inds, v, c, r2):
    match_dict = {}
    for i, k in zip(x1_inds, x2_inds):
        tup = (min(i,k), max(i,k))
        if not (tup in match_dict):

            match_dict[tup] = 1
            l = correct_center(v[i], v[k], c[i], c[k], r2 + 0.01)
            c_i = c[i] - l*v[i]
            c_k = c[k] - l*v[k]
            c[i, :] = c_i
            c[k, :] = c_k

            alpha = c[k, :] - c[i, :]
            alpha2 = alpha**2

            delta_alpha = alpha2[1] - alpha2[0]
            norm_alpha = alpha2[0] + alpha2[1]
            mult_alpha = alpha[0]*alpha[1]

            new_vx = (delta_alpha * v[i, 0] - 2*mult_alpha*v[i, 1])/norm_alpha
            new_vy = (-delta_alpha * v[i, 1] - 2*mult_alpha*v[i, 0])/norm_alpha
            v[i, 0] = new_vx
            v[i, 1] = new_vy

            new_vx = (delta_alpha * v[k, 0] - 2*mult_alpha*v[k, 1])/norm_alpha
            new_vy = (-delta_alpha * v[k, 1] - 2*mult_alpha*v[k, 0])/norm_alpha

            v[k, 0] = new_vx
            v[k, 1] = new_vy

    return v, c

if __name__ == "__main__":
    print("Test1: ", update_vectors(np.array([0]), np.array([1]), np.array([[2, 2], [-2, -2]],dtype=np.float64), np.array([[0, 0], [2, 4]])))
    print("Test2: ", update_vectors(np.array([0]), np.array([1]), np.array([[2, 2], [-2, -2]],dtype=np.float64), np.array([[0, 0], [4, 4]])))
    print("Test3: ", update_vectors(np.array([0]), np.array([1]), np.array([[1, 0], [0, 0]],dtype=np.float64), np.array([[0, 0], [4, 0]])))
    print("Test4: ", update_vectors(np.array([0, 1]), np.array([1, 2]), np.array([[2, 2], [-2, 2], [-2, -2]],dtype=np.float64), np.array([[0, 0], [4, 4], [8, 8]])))
    exit()

    for N in [200, 500, 1000, 10000, 20000]:
        x = np.arange(0, N, 2, dtype=int)
        y = np.arange(0, N, 2, dtype=int)
        centers = np.vstack((x, y)).T #Center points of circles
        velocities = np.vstack((np.random.normal(0, 1,size=N), np.random.normal(0, 1, size=N))).T

        t1 = time.time()
        inds1, inds2 = get_collisions(centers, 16)
        t2 = time.time()
        ret = update_vectors(inds1, inds2, velocities, centers)
        t3 = time.time()
        d1 = t2 - t1
        d2 = t3 - t2
        df = t3 - t1
        print("vectorize: ", N,df, d1/df, d2/df, (len(inds1)-len(x))/2)
