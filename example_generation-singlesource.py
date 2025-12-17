import numpy as np
import spatialscaper as ss
import os
from scipy.spatial import geometric_slerp
from pathlib import Path
import argparse
import librosa
from spatialscaper.utils import  traj_2_ir_idx, translate_origin

from sympy import pi, sin, cos, sqrt, acos, atan2

def asCartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r       = rthetaphi[0]
    theta   = rthetaphi[1]* np.pi/180 # to radian
    phi     = rthetaphi[2]* np.pi/180
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return [x,y,z]

def asSpherical(xyz):
    #takes list xyz (single coord)
    # x       = xyz[0]
    # y       = xyz[1]
    # z       = xyz[2]
    # r       =  np.sqrt(x*x + y*y + z*z)
    # theta   =  np.acos(z/r)*180/ np.pi #to degrees
    # phi     =  np.atan2(y,x)*180/ np.pi
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arcsin(z/r)*180/ np.pi #to degrees
    phi     =  np.arctan2(y,x)*180/ np.pi
    return [r,theta,phi]



#theta -> Elevation
#Phi -> Azimuth
def fibo_tessellation_sampling(N, r=1, const_theta=None, const_phi=None):
    xyz = []
    polar_rad = []
    polar_deg = []
    for n in range(N):
        if const_phi is None:
            phi_n = 2*np.pi*n*(1-(2/(1+np.sqrt(5))))#Azimuth
        else:
            phi_n = const_phi

        if const_theta is None:    
            theta_n = np.arccos(1 - (2*n/N))#Elevation
        else:
            theta_n = const_theta
            
        xyz_n = [r * np.cos(phi_n)*np.sin(theta_n), r * np.sin(phi_n)*np.sin(theta_n), r * np.cos(theta_n)]
        xyz.append(xyz_n)
        polar_rad.append([r, theta_n, phi_n])
        polar_deg.append([r, theta_n*180/np.pi % 360, phi_n*180/np.pi % 360])

    xyz = np.array(xyz)
    polar_rad = np.array(polar_rad)
    polar_deg = np.array(polar_deg)
    return xyz, polar_rad, polar_deg



# List of possible rooms to use for soundscape generation. Change 'ROOM' variable to one of these:
# "metu", "arni","bomb_shelter", "gym", "pb132", "pc226", "sa203", "sc203", "se203", "tb103", "tc352"
# Each room has a different Room Impulse Response (RIR) file associated with it, affecting the acoustic properties.

# FSD50K sound classes that will be spatialized include:
# 'femaleSpeech', 'maleSpeech', 'clapping', 'telephone', 'laughter',
# 'domesticSounds', 'footsteps', 'doorCupboard', 'music',
# 'musicInstrument', 'waterTap', 'bell', 'knock'.
# These classes are sourced from the FSD50K dataset, and
# are consistent with the DCASE SELD challenge classes.






#TAU_rooms = ['bomb_shelter', 'gym', 'pb132', 'pc226', 'sa203', 'sc203', 'se203', 'tb103', 'tc352']
SOUNDSPACES_rooms = ['ur6pFq6Qu1A']#['1LXtFkjw3qL'] #['ur6pFq6Qu1A']


# Constants
NSCAPES = 10  # Number of soundscapes to generate
FOREGROUND_DIR = "/home/pk3251/vast/DATA/spatial_eval_metrics/dataset/curated_single_source_stems"  # Directory with FSD50K foreground sound files
RIR_DIR = (
    #"/home/pk3251/scratch/appdir/Github/SpatialScaper/datasets/rir_datasets"  # Directory containing Room Impulse Response (RIR) files
    "/soundspaces/"
)
# ROOM = "gym"  # Initial room setting, change according to available rooms listed below
FORMAT = "foa"  # Output format specifier: could be 'mic' or 'foa'
N_EVENTS_MEAN = 1  # Mean number of foreground events in a soundscape
N_EVENTS_STD = 1  # Standard deviation of the number of foreground events
DURATION = 10.0  # Duration in seconds of each soundscape
SR = 16000  # SpatialScaper default sampling rate for the audio files
#OUTPUT_DIR = "/home/pk3251/vast/DATA/spatial_scaper_output_gym_azimuth_clap"  # Directory to store the generated soundscapes

REF_DB = (
    -65
)  # Reference decibel level for the background ambient noise. Try making this random too!

num_stems = 1 #"/home/pk3251/vast/DATA/localization_variations/azimuth_variations/single_source/<class_id>/<class_id_roomname_stem#>/"

# Function to generate a soundscape
def generate_soundscapes(experiment_name=None):

    



    if experiment_name is None:
        experiment_name = 'azimuth'

    OUTPUT_DIR = f"/home/pk3251/vast/DATA/localization_variations_soundspaces/{experiment_name}_variations/single_source/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    class_ids = os.listdir(FOREGROUND_DIR)
    class_ids = ['_m_04k94']#['_m_032s66']#['_m_0l15bq'] #class_ids[:1]  # For testing, limit to first class only
    
    for class_id in class_ids:
        print(f"Generating soundscapes for class ID: {class_id}")

        stems = os.listdir(os.path.join(FOREGROUND_DIR, class_id))

        selected_rooms = [SOUNDSPACES_rooms[a] for a in np.random.randint(0, len(SOUNDSPACES_rooms), num_stems)]
        selected_stems = [os.path.join(FOREGROUND_DIR, class_id, stems[a]) for a in np.random.randint(0, len(stems), num_stems)]

        #selected_stems = ['/home/pk3251/vast/DATA/spatial_eval_metrics/dataset/curated_single_source_stems/_m_0l15bq/21181_216.wav']
        selected_stems = ['/home/pk3251/vast/DATA/spatial_eval_metrics/dataset/curated_single_source_stems/_m_04k94/79223_169.wav']

        print(f"  Selected rooms: {selected_rooms}")
        print(f"  Selected stems: {selected_stems}")


        for stem_idx, (room, stem) in enumerate(zip(selected_rooms, selected_stems)):

            variation_dir = os.path.join(OUTPUT_DIR, class_id+'_'+room+'_'+Path(stem).stem+f"_{stem_idx+1}")
            print(f"*************************Stem {stem_idx}. Spatializing stem {Path(stem).stem} with room: {room}*************************")
            
            room_ir_path = os.path.join(RIR_DIR, room, "irs")
            # max_ir_len = 0
            # for ir in os.listdir(room_ir_path):
            #     aud, _ = librosa.load(os.path.join(room_ir_path, ir), sr=16000, mono=False)
            #     if aud.shape[1] > max_ir_len:
            #         max_ir_len = aud.shape[1]
            #         print('Max RIR length: ', max_ir_len, ir)

            # temp
            max_ir_len = 10722

            

            ####### CODE REPEATED TWICE FROM spatialscaper/core.py  -- TO BE MODULARISED LATER #######
            room_metadata_path = os.path.join( RIR_DIR, room, "points.txt")
            soundspaces_rir_listener_positions = np.array([[a[1],a[2],a[3]] for a in np.loadtxt(room_metadata_path, delimiter='\t')])

            # Calculate the center of the listener positions to use as the Actual Listener Position
            ssrir_origin_x = (np.min(soundspaces_rir_listener_positions.T[0])+np.max(soundspaces_rir_listener_positions.T[0]))/2
            ssrir_origin_y = (np.min(soundspaces_rir_listener_positions.T[1])+np.max(soundspaces_rir_listener_positions.T[1]))/2
            ssrir_origin_z = (np.min(soundspaces_rir_listener_positions.T[2])+np.max(soundspaces_rir_listener_positions.T[2]))/2
            ssrir_origin = [ssrir_origin_x, ssrir_origin_y, ssrir_origin_z]

            # Find the closest listener position in the SoundSpaces RIR dataset to the provided listener_position
            # if experiment_name == 'distance':
            #     ssrir_origin = [np.min(soundspaces_rir_listener_positions.T[0]), ssrir_origin_y, ssrir_origin_z]
            ss_selected_listener_index = traj_2_ir_idx(soundspaces_rir_listener_positions, [ssrir_origin])[0]
            ####### END OF REPEATED CODE #######

            ss_selected_listener_xyz = soundspaces_rir_listener_positions[ss_selected_listener_index]
            print('Listener Origin: ', ssrir_origin, ' | Selected Listener Position: ', soundspaces_rir_listener_positions[ss_selected_listener_index])
            
            for iscape in range(NSCAPES):
                ssc = ss.Scaper(
                    DURATION,
                    FOREGROUND_DIR,
                    RIR_DIR,
                    FORMAT,
                    room,
                    max_event_overlap=1,
                    max_event_dur=5.0,
                    speed_limit=2.0,  # in meters per second
                    DCASE_format=False,
                    sr=16000,
                )
                print(f"  Soundscape {iscape + 1}/{NSCAPES}")

                r, elevation, azimuth = None , None, None
                event_positions_ = None

                if experiment_name == 'distance':
                    r = np.linspace(1.0, 20.0, NSCAPES)
                    elevation = 90.0
                    azimuth = 0.0

                    print('Distances: ', r[iscape])

                    event_positions_ = asCartesian([r[iscape], elevation, azimuth])
                    print('-------------------------',[r[iscape], elevation, azimuth], event_positions_)
                    translated_event_positions_ = translate_origin(event_positions_, ss_selected_listener_xyz)
                    print('-------------------------',translated_event_positions_, asSpherical(translated_event_positions_))


                elif experiment_name == 'elevation':
                    r = 5.0
                    #elevation = np.linspace(85, -85, NSCAPES)
                    azimuth = 0.0

                    ele_xyz, ele_polar_rad, ele_polar_deg = fibo_tessellation_sampling(N=NSCAPES, r=5, const_phi=0.0)
                    ele_global = ele_polar_deg - 90#[np.argsort(ele_polar_deg[:,2])][:,2] - 180
                    elevation = ele_global[:,1]

                    print('Elevation angles: ', elevation)
                    event_positions_ = asCartesian([r, elevation[iscape], azimuth])
                    print('-------------------------',[r, elevation[iscape], azimuth], event_positions_)
                    translated_event_positions_ = translate_origin(event_positions_, ss_selected_listener_xyz)
                    print('-------------------------',translated_event_positions_)

                elif experiment_name == 'azimuth' or experiment_name is None:
                    r = 5.0
                    elevation = 90.0

                    azi_xyz, azi_polar_rad, azi_polar_deg = fibo_tessellation_sampling(N=NSCAPES, r=5, const_theta=0.5*np.pi)
                    azimuth_global = azi_polar_deg[np.argsort(azi_polar_deg[:,2])][:,2] - 180 # sort azimuth angles and range from -180 to 180
                    azimuth_global[0] = -175.0  #  to ensure first angle is exactly -175 degrees
                    
                    
                    azimuth = azimuth_global    
                    
                    event_positions_ = asCartesian([r, elevation, azimuth[iscape]])
                    translated_event_positions_ = translate_origin(event_positions_, ss_selected_listener_xyz)
                    print('-------------------------',translated_event_positions_)

                ssc.ref_db = REF_DB

                # static ambient noise
                #ssc.add_background()
            
                event_position_x = translated_event_positions_[0]
                event_position_y = translated_event_positions_[1]
                event_position_z = translated_event_positions_[2]

                ssc.add_event(label=('const',class_id), \
                source_file=('choose',[stem]),\
                source_time=('const',0.0001),\
                event_time=("const", 0.05),\
                event_position=("const", [[event_position_x, event_position_y, event_position_z]]),\
                snr=("uniform", 10, 10.001))  # randomly choosing and spatializing an FSD50K sound event
                

                track_name = class_id+'_'+room+f"_stem{stem_idx+1}" + f"_mix{iscape+1:03d}"
                audiofile = os.path.join(variation_dir, FORMAT, track_name)
                labelfile = os.path.join(variation_dir, "labels", track_name)

                ssc.generate(audiofile, labelfile, is_soundspaces=True, max_ir_len=max_ir_len)


def main():
    parser = argparse.ArgumentParser(
        description="Generate single-source soundscapes for a named experiment"
    )
    parser.add_argument(
        "experiment_name",
        nargs="?",
        default=None,
        help="Experiment name to select variation: 'azimuth', 'elevation', or 'distance' (default: azimuth)",
    )
    args = parser.parse_args()

    generate_soundscapes(args.experiment_name)


if __name__ == "__main__":
    main()
