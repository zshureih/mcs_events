import os
import cv2
import csv
import numpy as np
import vision.tracker.run_tracker as run_tracker
from vision.instSeg.inference import Solo_detector
from vision.utils import logger
import tensorflow as tf
import time
import glob
# def read_results(data_path,seq_name):
#     # seq_res = {
#     #   frame_id: {
#     #       'bbox': Nx4
#     #       'masks': NxWxH
#     #   }
#     seq_res = {}
#     gt_path = f"{data_path}/{seq_name}/gt.txt"
#     with open(gt_path) as det_file:
#         reader = csv.reader(det_file, delimiter=',')
#         for det_info in reader:
#             # 'frame_num', 0
#             # 'object_id', 1
#             # 'xmin', 'ymin', 'width', 'height', 2,3,4,5
#             # 'conf' 6
#             if int(det_info[6]) != 1:
#                 continue
#             frame_num = int(det_info[0])
#             xmin = int(np.maximum(0.0, float(det_info[2])))
#             ymin = int(np.maximum(0.0, float(det_info[3])))
#             width = int(float(det_info[4]))
#             height = int(float(det_info[5]))
#             if frame_num not in seq_res:
#                 seq_res[frame_num] = {
#                     'bboxes': [],
#                     'masks': []
#                 }
#             seq_res[frame_num]['bboxes'].append([xmin,ymin,width,height])
#     return seq_res

def read_seq(data_path,seq_name):
    seq = {
        'rgb': [],
        'depth': [],
        'mask': []
    }
    imgs_path = f"{data_path}/{seq_name}/RGB"
    num_frames = len(os.listdir(imgs_path))
    for frame_num in range(1,num_frames+1):
        rgb_path = os.path.join(imgs_path,str(int(frame_num)).zfill(6) + '.png')
        seq['rgb'].append(cv2.imread(rgb_path, cv2.COLOR_BGR2RGB))
        seq['depth'].append(cv2.imread(rgb_path.replace('RGB', 'Depth'))[:, :, 0])
        seq['mask'].append(cv2.imread(rgb_path.replace('RGB', 'Mask'))[:, :, ::-1])
    return seq 

def project_to_2D(points, rgb_image=None):
    if points.shape[1] == 3:
        points = np.hstack((points, np.ones((points.shape[0], 1))))
    # focal length
    # fi == (x * w) / u == (y * w) / v
    # or
    # fi_x == (x * w) / u
    # and
    # fi_y == (y * w) / v

    # offset parameters
    # x = ((fi_x * u) / w)) + d_x
    # y = ((fi_y * v) / w)) + d_y

    # skew parameters
    # x = ((fi_x*u + gamma*v) / w) + d_x
    # y = ((fi_y*v) / w) + d_y

    # position and orientation of camera
    # position w = (u,v,w)T of point in the world not expressed in the frame of reference of the camera
    # transform using 3d transformation
    # or w' = Omega * w + tau <=> point in frame of reference of the camera = transformation of point in frame of reference of world

    #intrinsic parameters: fi_x, fi_y, gamma, d_x, d_y
    # fi_x = fi_y
    # stored as
    fi = 500  # completely stumbled onto this
    CAM_ASPECT = [600, 400]
    cam_matrix = [
        [fi, 0, CAM_ASPECT[0] / 2, 0],
        [0, fi, CAM_ASPECT[1] / 2, 0],
        [0, 0, 1, 0],
    ]

    #extrinsic parameters: [omega | tau]
    # tau = [0, 1.5, -4.5] = camera position
    # perspective transform = diag([1,-1,1])
    extrinsic_matrix = [
        [1, 0, 0, 0],
        [0, -1, 0, 1.5],
        [0, 0, 1, 4.5],
        [0, 0, 0, 1]
    ]

    vert_pixels = []
    for vert in points:
        if vert[2] < 0:
            continue
        # vert = w_hat = world coord
        # O * w_hat
        omega_w_hat = np.dot(extrinsic_matrix, vert)
        # Delta * (O * w_hat)
        lambda_X = np.dot(cam_matrix, omega_w_hat)

        # x_hat = [lambda*x, lambda*y, lambda]
        # to convert back
        # x = x_hat / z_hat, y = y_hat / z_hat
        X = lambda_X / lambda_X[2]
        x, y, z = X
        x = int(x)
        y = int(y)

        # if the point is out of bounds, bound it to the image frame
        if x < 0:
            x = 0
        if x >= 600:
            x = 599
        if y < 0:
            y = 0
        if y >= 400:
            y = 399
        vert_pixels.append([x,y])
        # if point is out of bounds, skip it

        # rgb = list(obj['segment_color'].values())
        # # rgb_image = cv2.circle(rgb_image, (x, y), 1, rgb, thickness=2)
        # # cv2.imshow("winname", rgb_image)
        # # cv2.waitKey(1000)
        # # cv2.destroyAllWindows()

    vert_pixels = np.array(vert_pixels)
    return vert_pixels

def projectTo3D(obj_binary_mask, depth_image):
    # compute products of color as quick way to check for matches - these are unique in Oracle level metadata
    # mask_color = np.prod(obj_mask_color)
    # 1,2,3 | 0 = background
    # obj_masks = np.prod(mask_image, axis=-1)

    #get all pixels from the mask image
    obj_pixels = np.where(obj_binary_mask == 1)
    if len(obj_pixels[0]) == 0:
        return [], [], []
    
    pixel_depths = depth_image[obj_pixels]
    lambda_ws = []
    for i in range(len(obj_pixels[0])):
        y = obj_pixels[0][i] 
        x = obj_pixels[1][i] 
        lambda_ws.append([x,y,1])
    lambda_xs = np.array(lambda_ws)

    fi = 500  # figure why this works later
    CAM_ASPECT = [600, 400]
    cam_matrix = [
        [fi, 0, CAM_ASPECT[0] / 2],
        [0, fi, CAM_ASPECT[1] / 2],
        [0, 0, 1],
    ]
    inv_cam_matrix = np.linalg.inv(cam_matrix)

    extrinsic_matrix = [
        [1, 0, 0, 0],
        [0, -1, 0, 1.5],
        [0, 0, 1, 4.5],
        [0, 0, 0, 1]
    ]
    inv_world_matrix = np.linalg.inv(extrinsic_matrix)

    #convert image coords to camera coords
    cam_coords = np.dot(inv_cam_matrix, lambda_xs.T)
    # scale camera coords by depth
    # import pdb; pdb.set_trace()
    # depth_scaled = np.dot(np.diag(pixel_depths), cam_coords.T)
    depth_scaled = np.multiply(pixel_depths.reshape((pixel_depths.shape[0],1)), cam_coords.T)
    # assert (depth_scaled == depth_scaled2).all()
    # add homogeneous dimension to translate into world coords
    depth_scaled = np.append(depth_scaled, np.ones((depth_scaled.shape[0], 1)), axis=1)
    world_coords = np.dot(inv_world_matrix, depth_scaled.T)
    
    # get mean of all pixels in world coords for center of mass
    center_mass = world_coords.T.mean(0)

    # compute bounding box
    min_x, min_y, min_z, _ = np.min(world_coords.T, axis=0)
    max_x, max_y, max_z, _ = np.max(world_coords.T, axis=0)

    AABB = [max_x - min_x, max_y - min_y, max_z - min_z]
    centroid = [min_x + (AABB[0] / 2), min_y +(AABB[1] / 2), min_z + (AABB[2] / 2)]
    rotation = [0, 0, 0] # assume static initial rotation for now, pybullet doesn't really care too much
    return AABB, centroid, rotation   
# def get_obj_roles(predicted_tracks, depth_images, seq_res):
#     roles = []
#     # determine object parameters
#     # classify poles (done)
#     # anything that is connected to a pole but look like a wall is an occulder_wall
#         # go through poles
#         # see if something is connected
#         # if the max size of (w,h) the connected object is some occulder_wall setup, it is an occulder_wall
#     # anything that is stationary and look like a cube is a support object
#         # go through the missed classified objects
#         # check if something isn't stationary, it is a non-acotr
#         # else, if the max size of (w,h)
#     for obj_id, obj in predicted_tracks.items():
#         # if we're given shape, we have a simple heuristic to determine role
#         # if 'shape' in obj.keys():
#         #     init_step = min(obj['2d_position_history']['t'])
#         #     # if appears at beginning of scene and is cube
#         #     if init_step == 0 and (obj['shape'] == shape_keys['cube']):
#         #         obj['role'] = 'actor'
#         #     if obj['shape'] == shape_keys['pole']:
#         #         obj['role'] = 'actor'
#         #     if obj['shape'] == shape_keys['other'] or (init_step != 0 and obj['shape'] == shape_keys['cube']):
#         #         obj['role'] = 'non-actor'
        
#         # if we aren't given shape, determine roles from mask data
#         # else:
        
#         min_ys = []
#         min_xs = []
#         max_xs = []
#         largest_area = 0
#         dims = []
#         # get the history of each object
#         for k, t in enumerate(obj['2d_position_history']['t']):
#             # get the mask image at this step
#             masks = seq_res[t]['masks']
#             # get the obj's bbox at this point
#             x0, y0, w, h = obj['2d_position_history']['2dbbox'][k]
#             top_left = (x0, y0)

#             # get the mask from the detector results
#             masks = seq_res[t]['masks'][:, y0:y0+h, x0:x0+w]
#             # pick the channel in the masks w/ the largest sum
#             pick_size = np.sum(masks, axis=(1, 2))
#             obj_binary_mask = seq_res[t]['masks'][np.argmax(pick_size), :, :]
            
#             # keep track of the largest 2d bbox of the object
#             if w*h > largest_area:
#                 largest_area = w*h
#                 dims = [x0, y0, w, h]

#             min_y, max_y = top_left[1] - 1, min(top_left[1] + h + 1, 400)
#             min_x, max_x = top_left[0] - 1, min(top_left[0] + w + 1, 600)
#             # save these values for later
#             min_ys.append(min_y)
#             min_xs.append(min_x)
#             max_xs.append(max_x)
        
#         # save the largest bbox
#         obj['2d_dims'] = dims
        

#         # looking for pole objects
#         # if min value is near top, 
#         if sum(min_ys) <= 40 * len(min_ys) or sum(min_xs) <= 60 * len(min_xs) or sum(max_xs) == 600 * len(max_xs):
#             obj['role'] = 'actor'
#         else:  # looking for occluders
#             # grab the bbox at the initial timestep
#             bbox = obj['2d_position_history']['2dbbox'][0]
#             top_left = bbox[:2]
#             w, h = bbox[2], bbox[3]
#             min_y, max_y = top_left[1] - 1, min(top_left[1] + h + 1, depth_images[0].shape[0])
#             min_x, max_x = top_left[0] - 1, min(top_left[0] + w + 1, depth_images[0].shape[1])

#             # if the object is sufficiently near the middle of the screen and especially tall
#             if h > (2/5) * depth_images[0].shape[0]:
#                 obj['role'] = 'actor'
#             # TODO: look into numbers, look into which condition doesn't work

#         if 'role' not in obj.keys():
#             obj['role'] = 'non-actor'
def get_obj_roles(predicted_tracks):
        pole_objects = []
        for obj_id, obj in predicted_tracks.items():
            # first pass, look for poles
            # get all top_left points
            bboxes = obj['2d_position_history']['2dbbox']
            y_mins = [y for x, y, w, h in bboxes]
            x_mins = [x for x, y, w, h in bboxes]
            x_maxes = [x + w for x, y, w, h in bboxes]
            print(obj_id, 
                    sum(y_mins) <= 20 * len(y_mins), 
                    sum(x_mins) <= 30 * len(x_mins), 
                    sum(x_maxes) >= 580 * len(x_maxes))

            if sum(y_mins) <= 20 * len(y_mins) or sum(x_mins) <= 30 * len(x_mins) or sum(x_maxes) >= 570 * len(x_maxes):
                predicted_tracks[obj_id]['role'] = 'actor'
                pole_objects.append(obj_id)
        print(pole_objects)
        undetermined_objects = [
            obj_id 
            for obj_id in predicted_tracks.keys() 
            if obj_id not in pole_objects
        ]
        occluders = []
        support = []
        if len(pole_objects) != 0:
            for obj_id in pole_objects:
                obj = predicted_tracks[obj_id]
                init_step = min(obj['2d_position_history']['t'])
                # if the initial timestep is essentially 0
                if init_step <= 1:
                    print("stc, op4, sc")
                    # iterate through each potential object
                    for other_id in undetermined_objects:
                        # get the bbox at the initial step
                        bbox = predicted_tracks[other_id]['2d_position_history']['2dbbox'][0]
                        x, y, w, h = bbox

                        if h > (2/5) * 400:
                            predicted_tracks[other_id]['role'] = 'actor'
                            undetermined_objects.remove(other_id)
                            occluders.append(other_id)
                else:
                    print("grv")
                    # iterate through the objects
                    for other_id in undetermined_objects:
                        # if the object appeared in the first frame, it is an actor
                        if min(predicted_tracks[other_id]['2d_position_history']['t']) <= 1:
                            predicted_tracks[other_id]['role'] = 'actor'
                            undetermined_objects.remove(other_id)
                            support.append(other_id)
        
        for obj_id in undetermined_objects:
            predicted_tracks[obj_id]['role'] = 'non-actor'

        final_roles = []
        for obj_id, obj in predicted_tracks.items():
            print(obj_id, obj['role'])
            final_roles.append(obj['role'])

        roles, counts = np.unique(final_roles, return_counts=True)

# def display_results():
if __name__ == '__main__':
    # seqs = ["COL_0055_01"]
    # seqs_bad_pl = ["GRV_0051_01","GRV_0052_03","GRV_0053_01",
    # "GRV_0053_04","GRV_0056_03","GRV_0057_01",
    # "GRV_0057_04","GRV_0058_03","GRV_0061_02",
    # "GRV_0061_07","GRV_0067_01","GRV_0067_02",
    # "GRV_0067_07","GRV_0068_08","GRV_0069_04",
    # "GRV_0069_07","GRV_0069_08","GRV_0070_07",
    # "GRV_0070_08","GRV_0074_03","GRV_0074_06",
    # "GRV_0074_07","OP4_0060_01","OP4_0064_01",
    # "STC4_0051_02","STC4_0066_02","STC4_0071_01",
    # "STC4_0071_02","STC4_0074_01","STC4_0075_01",]
    # seqs_bad_imp = ["grav_24_0001_02","grav_76_0001_02",
    # "op_117_0001_01","op_122_0001_01","op_130_0001_01",
    # "op_183_0001_01","op_187_0001_02","op_196_0001_01",
    # "stc_358_0001_02",]  
    seqs = []
    # 
    # seqs.extend(seqs_bad_pl)
    # seqs.extend(seqs_bad_imp)
    # data_path = f"/scratch/alotaima/datasets/MCS/v6/test-det2"
    # data_path = "/nfs/hpc/share/alotaima/datasets/MCS/v5/test"
    data_path = "/nfs/hpc/share/alotaima/datasets/MCS/v7-test-100-each/data"

    # final_2
    # seqs = ["GRV_0074_08","OP4_0054_02","OP4_0055_02","OP4_0058_02","OP4_0060_01","OP4_0065_01","OP4_0067_01","OP4_0068_01","OP4_0068_02","OP4_0071_02","OP4_0073_01","SC_0053_01","SC_0054_01","SC_0064_01","SC_0064_02","SC_0070_02","SC_0071_02","STC4_0054_01","STC4_0057_02","STC4_0059_01","STC4_0068_02","STC4_0071_02",]
    # seqs = ["grav_52_0001_03","grav_97_0001_04","sc_204_0001_01","sc_219_0001_01","sc_232_0001_02","sc_240_0001_01","sc_256_0001_01","sc_292_0001_01","stc_302_0001_02","stc_308_0001_04","stc_311_0001_03","stc_318_0001_04","stc_321_0001_02","stc_339_0001_01","stc_345_0001_03","stc_361_0001_01","stc_377_0001_02","stc_388_0001_06","stc_396_0001_01","stc_400_0001_01",]
    # seqs = ["grav_19_0001_04","grav_32_0001_01","grav_33_0001_02","grav_34_0001_02","grav_39_0001_01","grav_58_0001_01","grav_61_0001_05","grav_65_0001_02","grav_72_0001_01","grav_7_0001_04","grav_83_0001_01","grav_84_0001_04","grav_85_0001_03","grav_8_0001_05","grav_92_0001_01","grav_95_0001_01","grav_96_0001_06","grav_98_0001_01","sc_202_0001_01","sc_205_0001_01","sc_209_0001_01","sc_212_0001_01","sc_215_0001_01","sc_221_0001_01","sc_226_0001_01","sc_228_0001_02","sc_230_0001_02","sc_234_0001_01","sc_235_0001_01","sc_236_0001_01","sc_238_0001_01","sc_241_0001_01","sc_242_0001_01","sc_243_0001_01","sc_244_0001_02","sc_246_0001_01","sc_247_0001_01","sc_248_0001_01","sc_250_0001_01","sc_255_0001_01","sc_258_0001_01","sc_259_0001_01","sc_260_0001_02","sc_261_0001_01","sc_262_0001_01","sc_267_0001_02","sc_269_0001_01","sc_270_0001_02","sc_273_0001_01","sc_274_0001_01","sc_275_0001_01","sc_276_0001_01","sc_277_0001_01","sc_279_0001_01","sc_281_0001_01","sc_282_0001_01","sc_284_0001_01","sc_286_0001_01","sc_287_0001_01","sc_288_0001_02","sc_295_0001_01","sc_296_0001_01","sc_298_0001_01","sc_300_0001_01","stc_310_0001_02","stc_319_0001_02","stc_323_0001_02","stc_324_0001_02","stc_328_0001_02","stc_331_0001_02","stc_334_0001_02","stc_335_0001_03","stc_338_0001_01","stc_340_0001_01","stc_350_0001_01","stc_351_0001_01","stc_353_0001_04","stc_360_0001_09","stc_363_0001_03","stc_366_0001_02","stc_367_0001_01","stc_374_0001_02","stc_379_0001_03","stc_381_0001_02","stc_383_0001_03","stc_384_0001_01","stc_386_0001_01","stc_393_0001_01","stc_394_0001_02","stc_398_0001_05",]
    # seqs = ["sc_209_0001_01"]#,"sc_243_0001_01","sc_247_0001_01","sc_248_0001_01","sc_255_0001_01","sc_267_0001_02","sc_295_0001_01","sc_296_0001_01",]
    seqs = ["sc_204_0001_01","sc_219_0001_01"]
    # files = glob.glob(f"{data_path}/*")
    # for file_ in files:
    #     seqs.append(file_.replace(f"{data_path}/",""))
    config   = 'vision/instSeg/configs/mcs4_fg/solov2_r101_dcn_fpn_8gpu_3x.py'
    weights  = 'vision/instSeg/solov2_mcs4_fg.pth'
    rgb = np.random.randint(0,255,(100000, 3))
    timer = {}
    print("init: running...")
    start = time.time()
    detector = Solo_detector(config=config, weights=weights)
    tracker = run_tracker.init()
    timer['init'] = time.time()-start
    print(f"init: completed! ({timer['init']})")
    # TODO: detections for frame 0 or tracker doesn't output results for frame 0
    output_path = './logger_output_final_l2_3_bad_look_into_results_with_z'
    if not os.path.isdir(f'{output_path}'):
        os.mkdir(f'{output_path}')
    for seq_name in seqs:
        try:
            print(f"start: {seq_name}")
            timer_seq = {}
            start_seq = time.time()
            # tf.reset_default_graph()
            seq = read_seq(data_path,seq_name)
            seq_res = {}
            print(seq_name)
            print("detector: running...")
            start = time.time()
            for frame_num, _ in enumerate(seq['rgb']):
                # seq_res[frame_num] = detector.step(seq['rgb'][frame_num], seq['depth'][frame_num])
                seq_res[frame_num] = detector.step(seq['rgb'][frame_num], seq['depth'][frame_num],seq['mask'][frame_num])
                # seq_res[frame_num] = get_detection()
            timer_seq['detector'] = time.time()-start
            print(f"detector: completed! ({timer_seq['detector']})")
            print("tracker: running...")
            start = time.time()
            results = run_tracker.main(seq_res,seq,tracker)
            timer_seq['tracker'] = time.time()-start
            print(f"tracker: completed! ({timer_seq['tracker']})")

            print("convertor: running...")
            start = time.time()
            for object_id in results.keys():
                results[object_id]['3d_position_history'] = {
                    "t": [],
                    "c": []
                }
                
                results[object_id]["3d_bounding_box_scale"] = [-1,-1,-1]
                for i, frame_num in enumerate(results[object_id]['2d_position_history']['t']):
                    x0,y0,w,h = results[object_id]['2d_position_history']['2dbbox'][i]
                    masks = seq_res[frame_num]['masks'][:, y0:y0+h, x0:x0+w]
                    pick_size = np.sum(masks,axis=(1,2))
                    obj_binary_mask = seq_res[frame_num]['masks'][np.argmax(pick_size), :, :] 
                    world_bbox, world_centroid, world_rotation = projectTo3D(obj_binary_mask, seq['depth'][frame_num])
                    results[object_id]['3d_position_history']['t'].append(frame_num)
                    results[object_id]['3d_position_history']['c'].append(world_centroid)
                    for i in range(3):
                        if results[object_id]["3d_bounding_box_scale"][i] < world_bbox[i]:
                            results[object_id]["3d_bounding_box_scale"][i] = world_bbox[i]
            timer_seq['convertor'] = time.time()-start
            print(f"convertor: completed! ({timer_seq['convertor']})")
            print("heurtistic: running...")
            start = time.time()
            get_obj_roles(results)
            timer_seq['heurtistic'] = time.time()-start
            print(f"heurtistic: completed! ({timer_seq['heurtistic']})")
            print("vis: running...")
            start = time.time()
            logger(seq,results,f"{output_path}/{seq_name}",rgb=rgb)
            timer_seq['vis'] = time.time()-start
            print(f"vis: completed! ({timer_seq['vis']})")
            timer_seq['total'] = time.time()-start_seq
            print(f"end: {seq_name} ({timer_seq['total']})")
            timer[seq_name] = timer_seq
        except OSError as err:
            print(err)
            print(seq_name)
            exit()
        
    # for k, v in timer.items():
    #     if type(v) == dict:
    #         print(f"seq: {k}")
    #         for stage_name, stage_time in v.items():
    #             print(f"{stage_name}: {stage_time}")
    #     else:
    #         print(f"{k}: {v}")
