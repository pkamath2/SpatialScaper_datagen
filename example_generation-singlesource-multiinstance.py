import numpy as np
import spatialscaper as ss
import os
from scipy.spatial import geometric_slerp
from pathlib import Path
import argparse

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
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.acos(z/r)*180/ np.pi #to degrees
    phi     =  np.atan2(y,x)*180/ np.pi
    return [r,theta,phi]
# List of possible rooms to use for soundscape generation. Change 'ROOM' variable to one of these:
# "metu", "arni","bomb_shelter", "gym", "pb132", "pc226", "sa203", "sc203", "se203", "tb103", "tc352"
# Each room has a different Room Impulse Response (RIR) file associated with it, affecting the acoustic properties.

# FSD50K sound classes that will be spatialized include:
# 'femaleSpeech', 'maleSpeech', 'clapping', 'telephone', 'laughter',
# 'domesticSounds', 'footsteps', 'doorCupboard', 'music',
# 'musicInstrument', 'waterTap', 'bell', 'knock'.
# These classes are sourced from the FSD50K dataset, and
# are consistent with the DCASE SELD challenge classes.






TAU_rooms = ['bomb_shelter', 'gym', 'pb132', 'pc226', 'sa203', 'sc203', 'se203', 'tb103', 'tc352']


# Constants
NSCAPES = 10  # Number of soundscapes to generate
FOREGROUND_DIR = "/home/pk3251/vast/DATA/spatial_eval_metrics/dataset/curated_single_source_stems"  # Directory with FSD50K foreground sound files
RIR_DIR = (
    "/home/pk3251/scratch/appdir/Github/SpatialScaper/datasets/rir_datasets"  # Directory containing Room Impulse Response (RIR) files
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

num_stems = 10 #"/home/pk3251/vast/DATA/localization_variations/azimuth_variations/single_source/<class_id>/<class_id_roomname_stem#>/"

# Function to generate a soundscape
def generate_soundscapes(experiment_name=None):

    if experiment_name is None:
        experiment_name = 'azimuth'

    OUTPUT_DIR = f"/home/pk3251/vast/DATA/localization_variations/{experiment_name}_variations/single_source_multiinstance/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    class_ids = os.listdir(FOREGROUND_DIR)
    # class_ids = ['_m_07plz5l']#class_ids[:1]  # For testing, limit to first class only
    
    for class_id in class_ids:
        print(f"Generating soundscapes for class ID: {class_id}")

        stems = os.listdir(os.path.join(FOREGROUND_DIR, class_id))

        selected_rooms = [TAU_rooms[a] for a in np.random.randint(0, len(TAU_rooms), num_stems)]
        selected_stems = [os.path.join(FOREGROUND_DIR, class_id, stems[a]) for a in np.random.randint(0, len(stems), num_stems)]

        print(f"  Selected rooms: {selected_rooms}")   
        print(f"  Selected stems: {selected_stems}")     

        for stem_idx, (room, stem) in enumerate(zip(selected_rooms, selected_stems)):
            variation_dir = os.path.join(OUTPUT_DIR, class_id+'_'+room+'_'+Path(stem).stem+f"_{stem_idx+1}")
            print(f"*************************Stem {stem_idx}. Spatializing stem {Path(stem).stem} with room: {room}*************************")
            for iscape in range(NSCAPES):
                print(f"  Soundscape {iscape + 1}/{NSCAPES}")
                ssc = ss.Scaper(
                    DURATION,
                    FOREGROUND_DIR,
                    RIR_DIR,
                    FORMAT,
                    room,
                    max_event_overlap=2,
                    max_event_dur=6.0,
                    speed_limit=2.0,  # in meters per second
                    DCASE_format=False,
                )

                ssc.ref_db = REF_DB

                # static ambient noise
                #ssc.add_background()

                r1, r2, elevation1, elevation2, azimuth1, azimuth2 = None , None, None, None , None, None
                event_positions1_, event_positions2_ = None, None

                if experiment_name == 'distance':
                    r1 = np.linspace(0.5, 5.0, NSCAPES)
                    elevation1 = 90.0
                    azimuth1 = 0.0

                    r2 = np.linspace(5.0, 0.5, NSCAPES)
                    elevation2 = 90.0
                    azimuth2 = 0.0

                    event_positions1_ = asCartesian([r1[iscape], elevation1, azimuth1])
                    event_positions2_ = asCartesian([r2[iscape], elevation2, azimuth2])

                elif experiment_name == 'elevation':
                    r1 = 5.0
                    elevation1 = np.linspace(85, -85, NSCAPES)
                    azimuth1 = 0.0

                    r2 = 5.0
                    elevation2 = np.linspace(-85, 85, NSCAPES)
                    azimuth2 = 0.0

                    event_positions1_ = asCartesian([r1, elevation1[iscape], azimuth1])
                    event_positions2_ = asCartesian([r2, elevation2[iscape], azimuth2])

                elif experiment_name == 'azimuth' or experiment_name is None:
                    r1 = 5.0
                    elevation1 = 90.0
                    azimuth1 = np.linspace(175, -175, NSCAPES) 

                    r2 = 5.0
                    elevation2 = 90.0
                    azimuth2 = np.linspace(-175, 175, NSCAPES) 

                    event_positions1_ = asCartesian([r1, elevation1, azimuth1[iscape]])
                    event_positions2_ = asCartesian([r2, elevation2, azimuth2[iscape]])
            
                event_position_x1 = event_positions1_[0]
                event_position_y1 = event_positions1_[1]
                event_position_z1 = event_positions1_[2]

                event_position_x2 = event_positions2_[0]
                event_position_y2 = event_positions2_[1]
                event_position_z2 = event_positions2_[2]
                # print(iscape, r, azimuth[iscape], elevation, event_position_x, event_position_y, event_position_z)

                ssc.add_event(label=('const',class_id), \
                source_file=('choose',[stem]),\
                source_time=('const',0.0001),\
                event_time=("const", 0.05),\
                event_position=("const", [[event_position_x1, event_position_y1, event_position_z1]]),\
                snr=("uniform", 10, 10.001))  # randomly choosing and spatializing an FSD50K sound event


                ssc.add_event(label=('const',class_id), \
                source_file=('choose',[stem]),\
                source_time=('const',0.0001),\
                event_time=("const", 3),\
                event_position=("const", [[event_position_x2, event_position_y2, event_position_z2]]),\
                snr=("uniform", 10, 10.001))  # randomly choosing and spatializing an FSD50K sound event
                

                track_name = class_id+'_'+room+f"_stem{stem_idx+1}_multiinstance" + f"_mix{iscape+1:03d}"
                audiofile = os.path.join(variation_dir, FORMAT, track_name)
                labelfile = os.path.join(variation_dir, "labels", track_name)

                ssc.generate(audiofile, labelfile)


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
