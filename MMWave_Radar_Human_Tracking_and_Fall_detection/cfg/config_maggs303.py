import config
# global save parameters
EXPERIMENT_NAME = 'test'
RADAR_FPS = 20  # 20 frames per second, 50ms per frame
CAMERA_FPS = 30  # 30 frames per second, lower under worse light condition
# manual save
MANSAVE_ENABLE = False  # this controls the flag from the source
MANSAVE_PERIOD = 1  # second, the time period saved for manual save
# auto save
AUTOSAVE_ENABLE = False  # auto save function requires tracking system
AUTOSAVE_PERIOD = 1 # second, the max time period saved for auto save (radar)

# multiple class instantiated, multiple config used
RADAR_CFG_LIST = [
    {'name'          : 'IWR1843_Ori',
     'cfg_port_name' : config.cfg0_port_name,
     'data_port_name': config.data0_port_name,
     'cfg_file_name' : './cfg/IWR1843_3D_20fps_15db.cfg',  # always use 3D data as input
     'xlim'          : None,  # the x-direction limit for cloud points from this single radar, set as [a, b), from radar view
     'ylim'          : (0.25, 6),
     'zlim'          : None,
     'pos_offset'    : (-0.8, 0, 2.22),  # default pos_offset is (0, 0, 0)
     'facing_angle'  : {'angle': (-30, 180, 0), 'sequence': None},  # right-hand global coord-sys, (x, y, z): [-180, 180] positive counted anti-clockwise when facing from axis end towards origin, default rotation sequence: zyx
     'ES_threshold'  : {'range': (0, None), 'speed_none_0_exception': True},  # if speed_none_0_exception is True, then the data with low ES but with speed will be reserved
     },

     {'name'          : 'IWR1843_Side',
     'cfg_port_name' : config.cfg1_port_name,
     'data_port_name': config.data1_port_name,
     'cfg_file_name' : './cfg/IWR1843_3D_20fps_15db.cfg',  # always use 3D data as input
     'xlim'          : None,  # the x-direction limit for cloud points from this single radar, set as [a, b), from radar view
     'ylim'          : (0.25, 6),
     'zlim'          : None,
     'pos_offset'    : (0.8, 4.81, 2.22),  # default pos_offset is (0, 0, 0)
     'facing_angle'  : {'angle': (30, 180, 180), 'sequence': None},  # right-hand global coord-sys, (x, y, z): [-180, 180] positive counted anti-clockwise when facing from axis end towards origin, default rotation sequence: zyx
     'ES_threshold'  : {'range': (0, None), 'speed_none_0_exception': True},  # if speed_none_0_exception is True, then the data with low ES but with speed will be reserved
     },
]

# multiple class instantiated, single config used
FRAME_EARLY_PROCESSOR_CFG = {  # early process config
    'FEP_frame_deque_length': 10,  # the number of frame stacked
    'FEP_background_noise_file': './data/03_09_2024_katie_pc_bg_noise/all_samples_merged.pkl',
    'FEP_background_removal_threshold': 0.1
}

# single class instantiated, single config used
VISUALIZER_CFG = {
    'dimension'               : '3D',  # only effect visualizer demo,
    'VIS_xlim'                : (-3, 3),
    'VIS_ylim'                : (0, 4.75),
    'VIS_zlim'                : (0, 2),

    'auto_inactive_skip_frame': int(1 * RADAR_FPS),  # frames, the time period will be skipped when auto is inactive
}

# single class instantiated, single config used
FRAME_POST_PROCESSOR_CFG = {  # post process config
    # cloud point filter para
    'FPP_global_xlim' : (-2, 2),  # the x-direction limit for merged cloud points from all radars, set as [a, b), from global view
    'FPP_global_ylim' : (0, 4),
    'FPP_global_zlim' : (0, 2),
    'FPP_ES_threshold': {'range': None, 'speed_none_0_exception': True},  # the points in this energy strength range will be preserved, if speed_none_0_exception is True, then the data with low ES but with speed will be reserved
}

# single class instantiated, single config used
DBSCAN_GENERATOR_CFG = {
    'Default'             : {
        'DBS_eps'        : 0.5,
        'DBS_min_samples': 10,

        # DBSCAN filter parameters
        'DBS_cp_pos_xlim': None,
        'DBS_cp_pos_ylim': None,
        'DBS_cp_pos_zlim': (0, 1.8),
        'DBS_size_xlim'  : (0.3, 1),
        'DBS_size_ylim'  : (0.3, 1),
        'DBS_size_zlim'  : (0.3, 2),
        'DBS_sort'       : None,
    },

    # More lenient settings for low energy levels
    'Dynamic_ES_0_above'  : {  # more lenient for data points with energy above 0
        'DBS_eps'        : 0.4,  # increased from 0.2
        'DBS_min_samples': 5,    # reduced from 15
    },
    'Dynamic_ES_100_above': {
        'DBS_eps'        : 0.5,  # increased from 0.3
        'DBS_min_samples': 4,    # reduced from 12
    },
    'Dynamic_ES_200_above': {
        'DBS_eps'        : 0.6,  # increased slightly from 0.4
        'DBS_min_samples': 6,
    },
    'Dynamic_ES_300_above': {
        'DBS_eps'        : 0.6,
        'DBS_min_samples': 3,
        'DBS_size_xlim'  : (0.1, 0.8),
        'DBS_size_ylim'  : (0.1, 0.8),
        'DBS_size_zlim'  : (0.1, 2),
    },
    'Dynamic_ES_400_above': {
        'DBS_eps'        : 1,
        'DBS_min_samples': 2,
        'DBS_size_xlim'  : (0.1, 0.8),
        'DBS_size_ylim'  : (0.1, 0.8),
        'DBS_size_zlim'  : (0.1, 2),
    },
}


# single class instantiated, single config used
BGNOISE_FILTER_CFG = {  # Background noise filter config
    'BGN_enable'             : False,

    # BGN processing para
    'BGN_deque_length'       : 150,
    'BGN_accept_ES_threshold': (None, 0),  # the noise with this ES range will be accepted when BGN update, it is for DBS noise
    'BGN_filter_ES_threshold': (None, 0),  # the noise with this ES range will be filtered when BGN filter
    'BGN_DBS_window_step'    : 20,
    'BGN_DBS_eps'            : 0,
    'BGN_DBS_min_samples'    : 0,
    'BGN_cluster_tf'         : 0.15,  # the threshold factor of data number used to select cluster
    'BGN_cluster_xextension' : 0.05,
    'BGN_cluster_yextension' : 0.05,
    'BGN_cluster_zextension' : 0.01,
}

# single class instantiated, single config used
HUMAN_TRACKING_CFG = {  # tracking system config
    'TRK_enable'                      : True,

    # Tracking system para
    'TRK_obj_bin_number'              : 2,  # the maximum number of object which can be detected
    'TRK_poss_clus_deque_length'      : 3,  # the number of possible clusters stacked before calculating the poss matrix
    'TRK_redundant_clus_remove_cp_dis': 0.8,  # the distance for remove redundant clusters closed to the updated one for multiple obj bin purpose
}

# multiple class instantiated, single config used
HUMAN_OBJECT_CFG = {  # human object config for each object bin
    'obj_deque_length'          : 60,  # the length of central point, size and status info in timeline stored in object bin

    # object update possibility config
    # related_possibility, hard limit
    'dis_diff_threshold'        : {
        'threshold'    : 0.8,  # the distance threshold(m) between the current cp and previous one for object info update
        'dynamic_ratio': 0.2,  # the speed ratio for dynamic distance threshold, 0-do nothing
    },
    'size_diff_threshold'       : 1,  # the size diff threshold(m^3) between the current cp and previous one for object info update
    # self_possibility, soft limit, (hard limit is set in DBSCAN_min/max_size), this is also used for classify the status
    'expect_pos'                : {
        'default' : (None, None, 1.1),
        'standing': (None, None, 1.1),
        'sitting' : (None, None, 0.7),
        'lying'   : (None, None, 0.5),
    },
    'expect_shape'              : {
        'default' : (0.8, 0.8, 1.8),
        'standing': (0.7, 0.7, 1.5),
        'sitting' : (0.3, 0.3, 0.6),
        'lying'   : (0.8, 0.8, 0.4),
    },
    'sub_possibility_proportion': (1, 1, 1.8, 1.2),  # the coefficient for the possibility proportion
    'inactive_timeout'          : 5,  # second, if timeout, object bin status goes inactive
    'obj_delete_timeout'        : 15,  # second, if timeout, delete this object bin

    # an entrance zone, for object bin start picking up an object
    'fuzzy_boundary_enter'      : False,
    'fuzzy_boundary_threshold'  : 0.5,
    'scene_xlim'                : FRAME_POST_PROCESSOR_CFG['FPP_global_xlim'],
    'scene_ylim'                : FRAME_POST_PROCESSOR_CFG['FPP_global_ylim'],
    'scene_zlim'                : FRAME_POST_PROCESSOR_CFG['FPP_global_zlim'],

    # object status threshold
    'standing_sitting_threshold': 0.8,
    'sitting_lying_threshold'   : 0.4,

    # get last update 2-5 info to show the current position and status
    'get_fuzzy_pos_No'          : 20,
    'get_fuzzy_status_No'       : 40,
}

# single class instantiated, single config used
SAVE_CENTER_CFG = {
    'file_save_dir'             : './data/Maggs_307/',
    'experiment_name'           : EXPERIMENT_NAME,
    # time saved in filename is the end time for manual mode
    # time saved in filename is the start time for auto mode

    # manual save
    'mansave_period'            : MANSAVE_PERIOD,  # the time period saved for manual save
    # for radar
    'mansave_rdr_enable'        : MANSAVE_ENABLE,  # not work if MANSAVE_ENABLE is False
    'mansave_rdr_frame_max'     : int(MANSAVE_PERIOD * RADAR_FPS * 1.2),  # *1.2 to guarantee the sequence integrity
    # for camera
    'mansave_cam_enable'        : MANSAVE_ENABLE,  # not work if MANSAVE_ENABLE is False
    'mansave_cam_frame_max'     : int(MANSAVE_PERIOD * CAMERA_FPS * 1.2),  # *1.2 to guarantee the sequence integrity

    # auto save
    # for radar
    'autosave_rdr_enable'       : AUTOSAVE_ENABLE,  # not work if AUTOSAVE_ENABLE is False
    'autosave_rdr_frame_max'    : int(AUTOSAVE_PERIOD * RADAR_FPS),
    'autosave_end_remove_period': HUMAN_OBJECT_CFG['obj_delete_timeout'] - HUMAN_OBJECT_CFG['inactive_timeout'],  # to remove the empty period for auto save
    # for camera
    'autosave_cam_enable'       : AUTOSAVE_ENABLE,  # not work if AUTOSAVE_ENABLE is False
    'autosave_cam_buffer'       : int((HUMAN_OBJECT_CFG['obj_delete_timeout'] - HUMAN_OBJECT_CFG['inactive_timeout']) * CAMERA_FPS / 2),
}

# single class instantiated, single config used
SYNC_MONITOR_CFG = {
    'rd_qsize_warning': 5,  # raw data queue size
    'sc_qsize_warning': 20,  # save center queue size
}
