"""
Auther: Zichao Shen
Date: 04/08/2023

In this project:
file_name is os.path.basename()
file_dir is os.path.dirname()
file_path is dir+name

All the limits are set as [a, b)
"""
import socket
import os
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tkinter as tk
import tkinter.font as tkf
from multiprocessing import Process, Manager
from time import sleep
import config
# import winsound

# import modules
from library import RadarReader, Visualizer, SyncMonitor
try:
    from library import SaveCenter
    SVC_enable = False
except:
    print("save center cannot be imported")
    pass
try:
    from library import Camera
    CAM_enable = False
except:
    pass
try:
    from library import EmailNotifier
    EMN_enable = False
except:
    pass
try:
    from library import VideoCompressor
    VDC_enable = False
except:
    pass

# import module configs
hostname = socket.gethostname()
print('Hostname is ' + hostname)
if hostname == 'IT077979RTX2080':
    from cfg.config_mvb501 import *
elif hostname == 'IT084378':
    from cfg.config_mvb340 import *
elif hostname == 'IT080027':
    from cfg.config_cp107 import *
elif hostname == config.hostname:
    from cfg.config_maggs303 import *
else: 
    raise Exception('Hostname is not found!')


def radar_proc_method(_run_flag, _radar_rd_queue, **_kwargs_CFG):
    radar = RadarReader(run_flag=_run_flag, radar_rd_queue=_radar_rd_queue, **_kwargs_CFG)
    radar.run()


def vis_proc_method(_run_flag, _radar_rd_queue_list, _shared_param_dict, **_kwargs_CFG):
    vis = Visualizer(run_flag=_run_flag, radar_rd_queue_list=_radar_rd_queue_list, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    vis.run()


def monitor_proc_method(_run_flag, _radar_rd_queue_list, _shared_param_dict, **_kwargs_CFG):
    sync = SyncMonitor(run_flag=_run_flag, radar_rd_queue_list=_radar_rd_queue_list, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    sync.run()


def save_proc_method(_run_flag, _shared_param_dict, **_kwargs_CFG):
    save = SaveCenter(run_flag=_run_flag, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    save.run()


def camera_proc_method(_run_flag, _shared_param_dict, **_kwargs_CFG):
    cam = Camera(run_flag=_run_flag, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    cam.run()


def email_proc_method(_run_flag, _shared_param_dict, **_kwargs_CFG):
    email = EmailNotifier(run_flag=_run_flag, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    email.run()


def vidcompress_proc_method(_run_flag, _shared_param_dict, **_kwargs_CFG):
    vidcompress = VideoCompressor(run_flag=_run_flag, shared_param_dict=_shared_param_dict, **_kwargs_CFG)
    vidcompress.run()


def test_proc_method(_run_flag, _shared_param_dict):
    _run_flag = _run_flag
    _auto_save_flag = _shared_param_dict['autosave_flag']

    def gui_button_style():
        style = tkf.Font(family='Calibri', size=16, weight=tkf.BOLD, underline=False, overstrike=False)
        return style

    def shot():
        if not _auto_save_flag.value:
           _auto_save_flag.value = True
           print("shot")
        else:
            _auto_save_flag.value = False

    def quit():
        _run_flag.value = False
        top.destroy()
        
    top = tk.Tk()
    top.wm_attributes('-topmost', 1)  # keep the window at the top
    top.overrideredirect(True)  # remove label area
    top.geometry('+30+30')
    top.resizable(False, False)
    #tk.Button(top, text='Capture', bg='lightblue', width=12, font=gui_button_style(), command=shot).grid(row=1, column=1)
    tk.Button(top, text='Quit', bg='lightblue', width=12, font=gui_button_style(), command=quit).grid(row=1, column=2)
    top.mainloop()


if __name__ == '__main__':
    # generate flag
    run_flag = Manager().Value('b', True)  # this flag control whole system running
    # generate save flag dict
    shared_param_dict = {'mansave_flag'       : Manager().Value('c', None),  # set as None, 'image' or 'video', only triggered at the end of recording
                         'autosave_flag'      : Manager().Value('b', True),  # set as False, True or False, constantly high from the beginning to the end of recording
                         'compress_video_file': Manager().Value('c', None),  # the record video file waiting to be compressed
                         'email_image'        : Manager().Value('f', None),  # for image info from save_center to email_notifier module
                         }

    # generate queues and essential processes
    radar_rd_queue_list = []  # radar rawdata queue list
    proc_list = []
    for RADAR_CFG in RADAR_CFG_LIST:
        radar_rd_queue = Manager().Queue() 
        kwargs_CFG = {'RADAR_CFG': RADAR_CFG, 'FRAME_EARLY_PROCESSOR_CFG': FRAME_EARLY_PROCESSOR_CFG}
        radar_proc = Process(target=radar_proc_method, args=(run_flag, radar_rd_queue), kwargs=kwargs_CFG)
        radar_rd_queue_list.append(radar_rd_queue)
        proc_list.append(radar_proc)
    kwargs_CFG = {'VISUALIZER_CFG'          : VISUALIZER_CFG,
                  'RADAR_CFG_LIST'          : RADAR_CFG_LIST,
                  'MANSAVE_ENABLE'          : MANSAVE_ENABLE,
                  'AUTOSAVE_ENABLE'         : AUTOSAVE_ENABLE,
                  'FRAME_POST_PROCESSOR_CFG': FRAME_POST_PROCESSOR_CFG,
                  'DBSCAN_GENERATOR_CFG'    : DBSCAN_GENERATOR_CFG,
                  'BGNOISE_FILTER_CFG'      : BGNOISE_FILTER_CFG,
                  'HUMAN_TRACKING_CFG'      : HUMAN_TRACKING_CFG,
                  'HUMAN_OBJECT_CFG'        : HUMAN_OBJECT_CFG,
                  'SYNC_MONITOR_CFG'        : SYNC_MONITOR_CFG}
    vis_proc = Process(target=vis_proc_method, args=(run_flag, radar_rd_queue_list, shared_param_dict), kwargs=kwargs_CFG)
    proc_list.append(vis_proc)
    monitor_proc = Process(target=monitor_proc_method, args=(run_flag, radar_rd_queue_list, shared_param_dict), kwargs=kwargs_CFG)
    proc_list.append(monitor_proc)
    

    # optional processes, can be disabled
    # try:
    #     kwargs_CFG.update({'SAVE_CENTER_CFG': SAVE_CENTER_CFG})
    #     shared_param_dict.update({'save_queue': Manager().Queue(maxsize=2000)})
    #     if SVC_enable:
    #         save_proc = Process(target=save_proc_method, args=(run_flag, shared_param_dict), kwargs=kwargs_CFG)
    #         proc_list.append(save_proc)
    # except:
    #     print("save center cannot be enabled")
    #     pass
    # try:
    #     kwargs_CFG.update({'CAMERA_CFG': CAMERA_CFG})
    #     if CAM_enable:
    #         camera_proc = Process(target=camera_proc_method, args=(run_flag, shared_param_dict), kwargs=kwargs_CFG)
    #         proc_list.append(camera_proc)
    # except:
    #     pass
    # try:
    #     kwargs_CFG.update({'EMAIL_NOTIFIER_CFG': EMAIL_NOTIFIER_CFG})
    #     if EMN_enable:
    #         email_proc = Process(target=email_proc_method, args=(run_flag, shared_param_dict), kwargs=kwargs_CFG)
    #         proc_list.append(email_proc)
    # except:
    #     pass
    # try:
    #     kwargs_CFG.update({'VIDEO_COMPRESSOR_CFG': VIDEO_COMPRESSOR_CFG})
    #     if VDC_enable:
    #         vidcompress_proc = Process(target=vidcompress_proc_method, args=(run_flag, shared_param_dict), kwargs=kwargs_CFG)
    #         proc_list.append(vidcompress_proc)
    # except:
    #     pass

    test_proc = Process(target=test_proc_method, args=(run_flag, shared_param_dict))
    proc_list.append(test_proc)

    # start the processes and wait to finish
    for t in proc_list:
        t.start()
        sleep(0.2)
    for t in proc_list:
        t.join()
        sleep(0.2)

    # winsound.Beep(1000, 500)
