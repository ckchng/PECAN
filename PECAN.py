import copy
import csv
import numpy as np

from src.get_data import *
from src.utils import *

from scipy.linalg import eig

# import matplotlib
# matplotlib.use('TkAgg')

from numba import njit, prange, cuda
from scipy.spatial import cKDTree

import csv
import time
from joblib import Parallel, delayed


def find_min_and_indices(min_obj_val, opt_pos_mat, opt_att_mat):
    """
    Find the minimum value and its indices from a given 2D matrix.
    Also, retrieve the corresponding opt_pos and opt_att matrices for those indices.

    Parameters:
    - min_obj_val: 2D matrix from which the minimum value and its indices are to be found
    - opt_pos_mat: Matrix corresponding to opt_pos
    - opt_att_mat: Matrix corresponding to opt_att

    Returns:
    - min_val: Minimum value from min_obj_val
    - min_indices: Indices of the minimum value in min_obj_val
    - opt_pos: Corresponding opt_pos matrix value
    - opt_att: Corresponding opt_att matrix value
    """
    min_val = np.min(min_obj_val)
    min_indices = np.unravel_index(np.argmin(min_obj_val), min_obj_val.shape)
    opt_pos = opt_pos_mat[min_indices]
    opt_att = opt_att_mat[min_indices]

    return min_val, min_indices, opt_pos, opt_att


def find_correspondences(CW_params, db_CW_params):
    # Initialize an empty list to store the indices of the correspondences
    correspondence_indices = []

    # Loop through each row in CW_params
    for i in range(CW_params.shape[0]):
        # Calculate the distance between the current row in CW_params and each row in db_CW_params
        distances = np.linalg.norm(db_CW_params - CW_params[i, :], axis=1)

        # Find the index of the minimum distance
        min_index = np.argmin(distances)

        # Append the index to the list of correspondences
        correspondence_indices.append(min_index)

    return np.array(correspondence_indices)




@njit
def compute_ellipse_distance_matrix(db_CW_conic_inv, db_CW_Hmi_k, P_mc, CC_params, neighbouring_craters_id):
    # Initialize the distance matrix
    pdist_mat = np.ones((CC_params.shape[0], len(neighbouring_craters_id))) * np.inf

    for ncidx, ncid in enumerate(neighbouring_craters_id):
        curr_db_CW_conic_inv = db_CW_conic_inv[ncid]

        # Project it down
        legit_flag, curr_A = conic_from_crater_cpu(curr_db_CW_conic_inv, db_CW_Hmi_k[ncid], P_mc)

        if not(legit_flag):
            continue

        # Extract xy first
        curr_A_params = extract_ellipse_parameters_from_conic(curr_A)

        if np.any(np.isnan(np.array(curr_A_params[1:]))):
            continue

        if not(curr_A_params[0]):
            continue

        for cc_id in range(CC_params.shape[0]):
            scaled_curr_A_params = np.array(curr_A_params[1:])
            pdist_mat[cc_id, ncidx] = np.linalg.norm(scaled_curr_A_params - CC_params[cc_id])

    return pdist_mat


def p1e_solver(CW_params, CW_ENU, Rw_c, CC_params, K, craters_id):
    # curr_CW_param = CW_params[craters_id]
    curr_CW_param = CW_params

    eps = 1
    Aell = np.diag([1 / curr_CW_param[3] ** 2, 1 / curr_CW_param[4] ** 2, 1 / eps ** 2])

    # Convert Aell to Aworld
    # Rw_ell = CW_ENU[craters_id]
    Rw_ell = CW_ENU

    # gt_att is the extrinsic, aka, it converts world's coord to cam's coord.
    Re_cam = Rw_c.T @ Rw_ell
    Acam = Re_cam @ Aell @ Re_cam.T

    # Backproject conic
    curr_conic = CC_params[craters_id]
    R_ellipse = np.zeros([2, 2])
    R_ellipse[0, 0] = np.cos(curr_conic[4])
    R_ellipse[0, 1] = -np.sin(curr_conic[4])
    R_ellipse[1, 0] = np.sin(curr_conic[4])
    R_ellipse[1, 1] = np.cos(curr_conic[4])
    R_ellipse = -R_ellipse
    Kc = np.linalg.inv(K) @ np.transpose(np.array([curr_conic[0], curr_conic[1], 1])) * K[0, 0]
    Ec = np.transpose(np.array([0, 0, 0]))

    # Compute Uc, Vc, Nc
    Uc = np.append(R_ellipse[:, 0], 0)
    Vc = np.append(R_ellipse[:, 1], 0)
    Nc = np.array([0, 0, 1])

    # Compute M
    M = np.outer(Uc, Uc) / curr_conic[2] ** 2 + np.outer(Vc, Vc) / curr_conic[3] ** 2

    # Compute W
    W = Nc / np.dot(Nc, (Kc - Ec))

    # Compute P
    P = np.identity(3) - np.outer((Kc - Ec), W)

    # Compute Q
    Q = np.outer(W, W)

    # Compute Bcam
    Bcam = P.T @ M @ P - Q

    V = eig(Acam, Bcam, left=True, right=False)[1]
    D = np.real(eig(Acam, Bcam)[0])

    sameValue, uniqueValue, uniqueIdx = differentiate_values(D)

    sigma_1 = uniqueValue
    sigma_2 = sameValue

    d1 = V[:, uniqueIdx].T
    d1 = d1 / np.linalg.norm(d1)
    k = np.sqrt(np.trace(np.linalg.inv(Acam)) - (1 / sigma_2) * np.trace(np.linalg.inv(Bcam)))

    delta_cam_est = k * d1
    delta_cam_est_flip = -k * d1

    delta_cam_world_est = Rw_c @ delta_cam_est.T
    E_w_est = curr_CW_param[0:3] + delta_cam_world_est

    delta_cam_world_flip_est = Rw_c @ delta_cam_est_flip.T
    E_w_flip_est = curr_CW_param[0:3] + delta_cam_world_flip_est

    return E_w_est, E_w_flip_est


def read_crater_database(craters_database_text_dir):
    with open(craters_database_text_dir, "r") as f:
        lines = f.readlines()[1:]  # ignore the first line
    lines = [i.split(',') for i in lines]
    lines = np.array(lines)

    ID = lines[:, 0]
    lines = np.float64(lines[:, 1:])

    db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, db_L_prime = get_craters_world(lines)
    crater_center_point_tree = cKDTree(db_CW_params[:, 0:3])
    return db_CW_params, db_CW_conic, db_CW_conic_inv, db_CW_ENU, db_CW_Hmi_k, ID, crater_center_point_tree, db_L_prime
    
    
    

def strip_symbols(s, symbols):
    for symbol in symbols:
        s = s.replace(symbol, '')
    return s



def testing_data_reading(dir):
    with open(dir, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = list(reader)

    camera_extrinsic = np.zeros([len(data), 3, 4])
    camera_pointing_angle = np.zeros(len(data))
    heights = np.zeros(len(data))
    noise_levels = np.zeros(len(data))
    remove_percentages = np.zeros(len(data))
    add_percentages = np.zeros(len(data))
    att_noises = np.zeros(len(data))  # Att_noise is always one value
    noisy_cam_orientations = np.zeros([len(data), 3, 3])  # Noisy cam orientation is always a 3x3 matrix

    imaged_params = []
    noisy_imaged_params = []
    crater_indices = []

    for row_id, row in enumerate(data):
        # Extract Camera Extrinsic matrix
        row_0 = row[0].split('\n')
        curr_cam_ext = np.zeros([3, 4])
        for i in range(len(row_0)):
            curr_row = strip_symbols(row_0[i], ['[', ']'])
            curr_array = np.array([float(value) for value in curr_row.split()]).reshape(1, 4)
            curr_cam_ext[i] = curr_array
        camera_extrinsic[row_id] = curr_cam_ext

        # Extract Camera Pointing Angle
        camera_pointing_angle[row_id] = float(row[1])

        # Extract Imaged Conics matrices
        curr_imaged_params = [np.array(conic) for conic in eval(row[2])]
        imaged_params.append(curr_imaged_params)

        # Extract Imaged Conics matrices
        curr_imaged_params = [np.array(conic) for conic in eval(row[3])]
        noisy_imaged_params.append(curr_imaged_params)

        # Extract Crater Indices
        # crater_indices.append(literal_eval(row[4]))
        curr_conic_indices = np.array(eval(row[4]))
        crater_indices.append(curr_conic_indices)

        # Extract Height
        heights[row_id] = float(row[5])

        # Extract Noise Level
        noise_levels[row_id] = float(row[6])

        # Extract Remove Percentage
        remove_percentages[row_id] = float(row[7])

        # Extract Add Percentage
        add_percentages[row_id] = float(row[8])

        # Extract Attitude Noise
        att_noises[row_id] = float(row[9])

        # Extract Noisy Camera Orientation
        # noisy_cam_orientations[row_id] = np.array(literal_eval(row[10]))
        row_10 = row[10].split('\n')
        curr_nc = np.zeros([3, 3])
        for i in range(len(row_10)):
            curr_row = strip_symbols(row_10[i], ['[', ']'])
            curr_array = np.array([float(value) for value in curr_row.split()]).reshape(1, 3)
            curr_nc[i] = curr_array
        noisy_cam_orientations[row_id] = curr_nc

    return camera_extrinsic, camera_pointing_angle, imaged_params, noisy_imaged_params, crater_indices, \
           heights, noise_levels, remove_percentages, add_percentages, att_noises, noisy_cam_orientations



def log_result(matched_ids, gt_ids, result_dir, i):

    # Initialize counts
    TP = FP = FN = TN = 0

    # Compute TP, FP, FN, and TN
    for m_id, gt_id in zip(matched_ids, gt_ids):
        if m_id != 'None' and gt_id != 'None':
            if m_id == gt_id:
                TP += 1
            else:
                FP += 1
        elif m_id == 'None' and gt_id != 'None':
            FN += 1
        elif m_id != 'None' and gt_id == 'None':
            FP += 1
        elif m_id == 'None' and gt_id == 'None':
            TN += 1

    # Compute rates
    FMR = FP / len([gt_id for gt_id in gt_ids])
    FNR = FN / len([gt_id for gt_id in gt_ids if gt_id != 'None'])
    
    matching_rate = TP / len([gt_id for gt_id in gt_ids if gt_id != 'None'])


    # Format the results in a single line
    result_str = ("Testing ID: {} | Matched IDs: {} | Matching Rate: {:.2f} | False Matching Rate: {:.2f} | False Negative Rate: {:.2f} \n").format(
        i, ', '.join(str(id) for id in matched_ids), matching_rate, FMR, FNR)

    # Open the file in append mode and write the result
    with open(result_dir, 'a') as file:
        file.write(result_str)



@njit
def find_nearest_neighbors(dist_matrix):
    M, N = dist_matrix.shape

    # Placeholder for nearest neighbors for each row
    nearest_neighbors_id = -np.ones(M, dtype=np.int32)
    nearest_neighbors_val = np.ones(M, dtype=np.float32) * np.inf

    # Flatten and argsort manually
    flat_size = M * N
    flat_distances = np.empty(flat_size, dtype=dist_matrix.dtype)
    for i in range(M):
        for j in range(N):
            flat_distances[i * N + j] = dist_matrix[i, j]

    sorted_indices = np.argsort(flat_distances)
    assigned_columns = set()
    for k in range(flat_size):
        index = sorted_indices[k]
        i = index // N
        j = index % N

        if nearest_neighbors_id[i] == -1 and j not in assigned_columns:
            nearest_neighbors_id[i] = j
            nearest_neighbors_val[i] = dist_matrix[i, j]
            assigned_columns.add(j)

        # # If there are more rows than columns, a column can be chosen multiple times.
        # # Otherwise, a column should be chosen at most once.
        # if nearest_neighbors_id[i] == -1 and (M > N or not np.any(nearest_neighbors_id == j)):
        #     nearest_neighbors_id[i] = j
        #     nearest_neighbors_val[i] = dist_matrix[i, j]

        # Break when all rows have been assigned
        if not np.any(nearest_neighbors_id == -1):
            break

    return nearest_neighbors_id, nearest_neighbors_val


def main_func(db_cw_id, CW_params, CW_ENU, CW_L_prime, Rw_c, CC_params, K, cc_id, gt_att,
              CW_conic_inv, CW_Hmi_k,
              px_thres, ab_thres, deg_thres,
              eld_thres, img_w, img_h):

    opt_num_matches = 0
    opt_cam_pos = np.array([0, 0, 0])
    opt_matched_ids = np.zeros(CC_params.shape[0])

    E_w_est, E_w_flip_est = p1e_solver(CW_params[db_cw_id], CW_L_prime[db_cw_id], Rw_c, CC_params, K,
                                       cc_id)

    rc_pos = gt_att @ -E_w_est
    rc_neg = gt_att @ -E_w_flip_est

    so3_pos = np.zeros([3, 4])
    so3_neg = np.zeros([3, 4])

    so3_pos[:, 0:3] = gt_att
    so3_neg[:, 0:3] = gt_att

    so3_pos[:, 3] = rc_pos
    so3_neg[:, 3] = rc_neg

    P_mc_pos = K @ so3_pos
    P_mc_neg = K @ so3_neg

    # chirality test here
    curr_crater_center_homo = np.array([CW_params[db_cw_id, 0:3]])
    curr_crater_center_homo = np.append(curr_crater_center_homo, 1)
    proj_pos = P_mc_pos @ curr_crater_center_homo.T
    proj_neg = P_mc_neg @ curr_crater_center_homo.T

    if proj_pos[2] > 0:  # chiraility test
        P_mc = P_mc_pos
        so3 = so3_pos
        cam_pos = E_w_est
    elif proj_neg[2] > 0:
        P_mc = P_mc_neg
        so3 = so3_neg
        cam_pos = E_w_flip_est
    else:
        return opt_num_matches, opt_matched_ids, opt_cam_pos
        # continue

    # compute the distance here
    legit_flag, curr_A = conic_from_crater_cpu(CW_conic_inv[db_cw_id], CW_Hmi_k[db_cw_id], P_mc)
    # Extract xy first
    if not (legit_flag):
        return opt_num_matches, opt_matched_ids, opt_cam_pos
        # continue

    curr_A_params = extract_ellipse_parameters_from_conic(curr_A)

    nextStageFlag = False
    # a pre-screen test here, the thresholds are set to be loose so that this step allows more pairs to be passed
    if curr_A_params[0]:
        px_dev = np.linalg.norm((curr_A_params[1:3]) - (CC_params[cc_id, 0:2]))
        a_dev = np.abs(curr_A_params[3] - CC_params[cc_id, 2]) / CC_params[cc_id, 2]
        b_dev = np.abs(curr_A_params[4] - CC_params[cc_id, 3]) / CC_params[cc_id, 3]
        phi_dev = np.abs(curr_A_params[-1] - CC_params[cc_id, -1])

        if px_dev < px_thres and a_dev < ab_thres and b_dev < ab_thres and phi_dev < np.radians(deg_thres):
            nextStageFlag = True

    if not (nextStageFlag):
        return opt_num_matches, opt_matched_ids, opt_cam_pos
        # continue

    # Only process the rest if the pre-screen test is passed
    neighbouring_craters_id = np.arange(CW_params.shape[0])

    # 1) project all 3D points onto the image plane
    projected_3D_points = P_mc @ np.hstack(
        [CW_params[neighbouring_craters_id, 0:3], np.ones((len(neighbouring_craters_id), 1))]).T
    points_on_img_plane = np.array([projected_3D_points[0, :] / projected_3D_points[2, :],
                                    projected_3D_points[1, :] / projected_3D_points[2, :]])

    within_img_valid_indices = np.where((points_on_img_plane[0, :] >= 0) &
                                        (points_on_img_plane[0, :] <= img_w) &
                                        (points_on_img_plane[1, :] >= 0) &
                                        (points_on_img_plane[1, :] <= img_h) &
                                        ~np.isnan(points_on_img_plane[0, :]) &
                                        ~np.isnan(points_on_img_plane[1, :]))[0]

    fil_ncid = neighbouring_craters_id[within_img_valid_indices]

    # check if the crater is visible to the camera
    _, fil_ncid = visible_points_on_sphere(CW_params[:, 0:3], np.array([0, 0, 0]),
                                           np.linalg.norm(CW_params[0, 0:3]),
                                           cam_pos, fil_ncid)

    if len(fil_ncid) == 0:
        return opt_num_matches, opt_matched_ids, opt_cam_pos

    try:
        el_dist_mat = compute_ellipse_distance_matrix(CW_conic_inv, CW_Hmi_k, P_mc, CC_params,
                                                      fil_ncid)
    except:
        el_dist_mat = np.ones((CC_params.shape[0], len(neighbouring_craters_id))) * np.inf

    nearest_neighbors_idx, nearest_neighbors_val = find_nearest_neighbors(el_dist_mat)
    closest_neighbouring_ids = [fil_ncid[idx] for idx in nearest_neighbors_idx]

    # first level test, if it passes, go to second level
    matched_count = np.sum(nearest_neighbors_val <= eld_thres)
    if not (matched_count > lower_matched_percentage * CC_params.shape[0]):
        return opt_num_matches, opt_matched_ids, opt_cam_pos
        # continue

    # Below is the refinement step to maximize the number of matches
    len_CC_params = CC_params.shape[0]
    CW_matched_ids = []
    CW_params_sub = np.zeros([len_CC_params, CW_params.shape[1]])
    CW_ENU_sub = np.zeros([len_CC_params, CW_ENU.shape[1], CW_ENU.shape[2]])
    CW_L_prime_sub = np.zeros([len_CC_params, CW_L_prime.shape[1], CW_L_prime.shape[2]])

    for j in range(CC_params.shape[0]):
        CW_matched_ids.append(ID[closest_neighbouring_ids[j]])
        CW_params_sub[j] = CW_params[closest_neighbouring_ids[j]]
        CW_ENU_sub[j] = CW_ENU[closest_neighbouring_ids[j]]
        CW_L_prime_sub[j] = CW_L_prime[closest_neighbouring_ids[j]]

    opt_num_matches, opt_matched_ids, opt_cam_pos = refinement(CW_params, CW_conic_inv, CW_Hmi_k, ID,
                                                                    CW_params_sub, CW_L_prime_sub,
                                                                    Rw_c, CC_params, K,
                                                                    eld_thres,
                                                                    img_w, img_h)

    return opt_num_matches, opt_matched_ids, opt_cam_pos


def refinement(CW_params, CW_conic_inv, CW_Hmi_k, ID,
                    CW_params_sub, CW_L_prime_sub,
                    Rw_c, CC_params, K, eld_thres,
                    img_w, img_h):
    '''
    #### here we look for the pair that leads to the highest consensus matches
    :param CW_params_sub: CW correspondence for CC_params
    :param Rw_c:
    :param CC_params:
    :param K:
    :return:
    '''
    gt_att = Rw_c.T
    so3_pos = np.zeros([3, 4])
    so3_neg = np.zeros([3, 4])

    so3_pos[:, 0:3] = gt_att
    so3_neg[:, 0:3] = gt_att
    max_match_count = 0
    opt_matched_ids = [[] for _ in range(CC_params.shape[0])]
    opt_cam_pos = np.array([0, 0, 0])
    for cc_id in range(CC_params.shape[0]):
        E_w_est, E_w_flip_est = p1e_solver(CW_params_sub[cc_id], CW_L_prime_sub[cc_id], Rw_c, CC_params, K,
                                           cc_id)

        rc_pos = gt_att @ -E_w_est
        rc_neg = gt_att @ -E_w_flip_est

        so3_pos[:, 3] = rc_pos
        so3_neg[:, 3] = rc_neg

        P_mc_pos = K @ so3_pos
        P_mc_neg = K @ so3_neg

        # chirality test here
        curr_crater_center_homo = np.array([CW_params_sub[cc_id, 0:3]])
        curr_crater_center_homo = np.append(curr_crater_center_homo, 1)

        proj_pos = P_mc_pos @ curr_crater_center_homo.T
        proj_neg = P_mc_neg @ curr_crater_center_homo.T

        if proj_pos[2] > 0:  # chiraility test
            P_mc = P_mc_pos
            so3 = so3_pos
            cam_pos = E_w_est
        elif proj_neg[2] > 0:
            P_mc = P_mc_neg
            so3 = so3_neg
            cam_pos = E_w_flip_est
        else:
            continue

        ##################### extract new craters
        neighbouring_craters_id = np.arange(CW_params.shape[0])
        # 1) project all 3D points onto the image plane
        projected_3D_points = P_mc @ np.hstack(
            [CW_params[neighbouring_craters_id, 0:3], np.ones((len(neighbouring_craters_id), 1))]).T
        points_on_img_plane = np.array([projected_3D_points[0, :] / projected_3D_points[2, :],
                                        projected_3D_points[1, :] / projected_3D_points[2, :]])

        within_img_valid_indices = np.where((points_on_img_plane[0, :] >= 0) &
                                            (points_on_img_plane[0, :] <= img_w) &
                                            (points_on_img_plane[1, :] >= 0) &
                                            (points_on_img_plane[1, :] <= img_h) &
                                            ~np.isnan(points_on_img_plane[0, :]) &
                                            ~np.isnan(points_on_img_plane[1, :]))[0]

        fil_ncid = neighbouring_craters_id[within_img_valid_indices]

        # TODO: check if the crater is visible to the camera
        _, fil_ncid = visible_points_on_sphere(CW_params[:, 0:3], np.array([0, 0, 0]),
                                               np.linalg.norm(CW_params[0, 0:3]),
                                               cam_pos, fil_ncid)

        if len(fil_ncid) == 0:
            continue

        try:
            el_dist_mat = compute_ellipse_distance_matrix(CW_conic_inv, CW_Hmi_k, P_mc, CC_params,
                                                          fil_ncid)
        except:
            el_dist_mat = np.ones((CC_params.shape[0], len(neighbouring_craters_id))) * np.inf

        nearest_neighbors_idx, nearest_neighbors_val = find_nearest_neighbors(el_dist_mat)
        closest_neighbouring_ids = [fil_ncid[idx] for idx in nearest_neighbors_idx]

        # nn_val_stack.append(nearest_neighbors_val)
        matched_count = 0
        matched_ids = [[] for _ in range(CC_params.shape[0])]

        for j in range(CC_params.shape[0]):
            if nearest_neighbors_val[j] <= eld_thres:
                matched_ids[j] = ID[closest_neighbouring_ids[j]]
                matched_count = matched_count + 1
            else:
                matched_ids[j] = 'None'

        if matched_count > max_match_count:
            max_match_count = copy.deepcopy(matched_count)
            opt_matched_ids = copy.deepcopy(matched_ids)
            opt_cam_pos = copy.deepcopy(cam_pos)

    return max_match_count, opt_matched_ids, opt_cam_pos


def visible_points_on_sphere(points, sphere_center, sphere_radius, camera_position, valid_indices):
    """Return the subset of the 3D points on the sphere that are visible to the camera."""
    visible_points = []
    visible_indices = []
    visible_len_P_cam = []

    for idx in valid_indices:
        point = points[idx, :]

        # 1. Translate the origin to the camera
        P_cam = point - camera_position

        # 2. Normalize the translated point
        P_normalized = P_cam / np.linalg.norm(P_cam)

        # 3 & 4. Solve for the real roots
        # Coefficients for the quadratic equation
        a = np.dot(P_normalized, P_normalized)
        b = 2 * np.dot(P_normalized, camera_position - sphere_center)
        c = np.dot(camera_position - sphere_center, camera_position - sphere_center) - sphere_radius ** 2

        discriminant = b ** 2 - 4 * a * c
        root1 = (-b + np.sqrt(discriminant)) / (2 * a)
        root2 = (-b - np.sqrt(discriminant)) / (2 * a)

        min_root = np.minimum(root1, root2)
        # 5. Check which real root matches the length of P_cam
        length_P_cam = np.linalg.norm(P_cam)

        # 6 & 7. Check visibility
        if (np.abs(min_root - length_P_cam) < 1000):
            visible_points.append(point)
            visible_indices.append(idx)
            visible_len_P_cam.append(length_P_cam)


    return visible_points, visible_indices

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to process data.")
    parser.add_argument("--data_dir", required=True, help="Directory path")
    parser.add_argument("--result_dir", required=True, help="Output path")
    parser.add_argument("--testing_data_dir", required=True, help="Testing path")
    parser.add_argument("--starting_id", type=int, required=True, help="starting_id")
    parser.add_argument("--step", type=int, required=True, help="step")
    parser.add_argument("--img_w", type=int, required=True, help="img_w")
    parser.add_argument("--img_h", type=int, required=True, help="img_h")
    parser.add_argument("--px_thres", type=float, required=True, help="px_thres")
    parser.add_argument("--deg_thres", type=float, required=True, help="rad_thres")
    parser.add_argument("--ab_thres", type=float, required=True, help="ab_thres")
    parser.add_argument("--epsilon", type=float, required=True, help="eld_thres")
    parser.add_argument("--lower_alpha", type=float, required=True, help="lower_matched_percentage")
    parser.add_argument("--alpha", type=float, required=True, help="upper_matched_percentage")
    parser.add_argument("--num_cores", type=int, required=True, help="starting_id")
    
    args = parser.parse_args()
    #
    data_dir = args.data_dir
    result_dir = args.result_dir
    testing_data_dir = args.testing_data_dir
    img_w = args.img_w
    img_h = args.img_h
    
    lower_matched_percentage = args.lower_alpha
    upper_matched_percentage = args.alpha
    num_cores = args.num_cores
    eld_thres = args.epsilon
    px_thres = args.px_thres
    ab_thres = args.ab_thres
    rad_thres = args.deg_thres
    
    starting_id = args.starting_id
    step = args.step
    ending_id = starting_id + step

    ### Read the craters database in raw form
    all_craters_database_text_dir = data_dir + '/robbins_navigation_dataset.txt'

    CW_params, CW_conic, CW_conic_inv, CW_ENU, CW_Hmi_k, ID, crater_center_point_tree, CW_L_prime = \
            read_crater_database(all_craters_database_text_dir)

    calibration_file = data_dir + '/calibration.txt'
    K = get_intrinsic(calibration_file)

    camera_extrinsic, camera_pointing_angle, imaged_params, noisy_imaged_params, craters_indices, \
    heights, noise_levels, remove_percentages, add_percentages, att_noises, noisy_cam_orientations = testing_data_reading(
        testing_data_dir)

    for i in range(starting_id, ending_id):

        cam = camera_extrinsic[i]
        noisy_att = noisy_cam_orientations[i]
        gt_pos = -cam[0:3, 0:3].T @ cam[0:3, 3]
        
        gt_att = noisy_att.T

        gt_ids = craters_indices[i]

        curr_img_params = noisy_imaged_params[i]

        start_time = time.time()  ###################### start time ####################################

        curr_craters_id = np.array(craters_indices[i])
        CC_params = np.zeros([len(curr_img_params), 5])
        CC_a = np.zeros([len(curr_img_params)])
        CC_b = np.zeros([len(curr_img_params)])
        CC_conics = np.zeros([len(curr_img_params), 3, 3])
        sigma_sqr = np.zeros([len(curr_img_params)])
        matched_idx = np.zeros([len(curr_img_params)])
        matched_ids = [[] for _ in range(len(curr_img_params))]
        ncp_match_flag = False
        cp_match_flag = False
        # Convert curr_img_params to CC_conics and compute sigma_sqr
        for j, param in enumerate(curr_img_params):
            CC_params[j] = param
            CC_conics[j] = ellipse_to_conic_matrix(*param)

        # sort CC_params based on size # bigger craters, better it is.
        sorted_indices = np.argsort(CC_params[:, 2])
        sorted_indices = sorted_indices[::-1]
        CC_params = CC_params[sorted_indices]

        CC_conics = CC_conics[sorted_indices]
        curr_craters_id = curr_craters_id[sorted_indices]

        Rw_c = gt_att.T
        # compute first Acam

        found_flag = False

        for cc_id in range(CC_params.shape[0]):
            results = Parallel(n_jobs=num_cores)(
                delayed(main_func)(
                    db_cw_id, CW_params, CW_ENU, CW_L_prime, Rw_c, CC_params, K, cc_id, gt_att,
                    CW_conic_inv, CW_Hmi_k,
                    px_thres, ab_thres, rad_thres,
                    eld_thres, img_w, img_h) for db_cw_id in range(CW_params.shape[0])
            )
            opt_num_matches_vec = [opt_num_matches[0] for opt_num_matches in results]

            # if there is one that's larger than
            if (np.max(opt_num_matches_vec) > upper_matched_percentage * CC_params.shape[0]):
                found_flag = True
                opt_matched_ids = results[np.argmax(opt_num_matches_vec)][1]
                opt_cam_pos = results[np.argmax(opt_num_matches_vec)][2]
                break

            if found_flag:
                break

        if not(found_flag):
            for j in range(CC_params.shape[0]):
                matched_ids[j] = 'None'
            opt_cam_pos = np.array([0, 0, 0])
            log_result(matched_ids, curr_craters_id, result_dir, i)
        else:
            log_result(opt_matched_ids, curr_craters_id,result_dir, i)

