import numpy as np
import spatialscaper as ss
import os
from scipy.spatial import geometric_slerp

from sympy import pi, sin, cos, sqrt, acos, atan2

# Constants
NSCAPES = 20  # Number of soundscapes to generate
FOREGROUND_DIR = "/home/pk3251/scratch/appdir/Github/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA"  # Directory with FSD50K foreground sound files
RIR_DIR = (
    "/home/pk3251/scratch/appdir/Github/SpatialScaper/datasets/rir_datasets"  # Directory containing Room Impulse Response (RIR) files
)
ROOM = "gym"  # Initial room setting, change according to available rooms listed below
FORMAT = "foa"  # Output format specifier: could be 'mic' or 'foa'
N_EVENTS_MEAN = 1  # Mean number of foreground events in a soundscape
N_EVENTS_STD = 1  # Standard deviation of the number of foreground events
DURATION = 2.0  # Duration in seconds of each soundscape
SR = 24000  # SpatialScaper default sampling rate for the audio files
#OUTPUT_DIR = "/home/pk3251/vast/DATA/spatial_scaper_output_gym_azimuth_clap"  # Directory to store the generated soundscapes
OUTPUT_DIR = "/home/pk3251/vast/DATA/test_withoutsnr_and_irnorm"
REF_DB = (
    -65
)  # Reference decibel level for the background ambient noise. Try making this random too!

# List of possible rooms to use for soundscape generation. Change 'ROOM' variable to one of these:
# "metu", "arni","bomb_shelter", "gym", "pb132", "pc226", "sa203", "sc203", "se203", "tb103", "tc352"
# Each room has a different Room Impulse Response (RIR) file associated with it, affecting the acoustic properties.

# FSD50K sound classes that will be spatialized include:
# 'femaleSpeech', 'maleSpeech', 'clapping', 'telephone', 'laughter',
# 'domesticSounds', 'footsteps', 'doorCupboard', 'music',
# 'musicInstrument', 'waterTap', 'bell', 'knock'.
# These classes are sourced from the FSD50K dataset, and
# are consistent with the DCASE SELD challenge classes.

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

# Function to generate a soundscape
def generate_soundscape(index):
    track_name = f"fold1_room1_mix{index+1:03d}"
    # Initialize Scaper. 'max_event_overlap' controls the maximum number of overlapping sound events.
    ssc = ss.Scaper(
        DURATION,
        FOREGROUND_DIR,
        RIR_DIR,
        FORMAT,
        ROOM,
        max_event_overlap=1,
        speed_limit=2.0,  # in meters per second
    )
    ssc.ref_db = REF_DB

    # static ambient noise
    ssc.add_background()

    # Add a random number of foreground events, based on the specified mean and standard deviation.
    n_events = int(np.random.normal(N_EVENTS_MEAN, N_EVENTS_STD))
    n_events = n_events if n_events > 0 else 1  # n_events should be greater than zero

    n_events = 1 

    r = 5.0
    elevation = 90.0
    azimuth = np.linspace(180, -180, 20) #Azimuthal angle0.0
    
    for _ in range(n_events):

        event_positions_ = asCartesian([r, elevation, azimuth[iscape]])#asCartesian([r, theta[iscape], phi])
        
        event_position_x = event_positions_[0]
        event_position_y = event_positions_[1]
        event_position_z = event_positions_[2]
        print(iscape, r, azimuth[iscape], elevation, event_position_x, event_position_y, event_position_z)

        ssc.add_event(label=('const','clapping'), \
        source_file=('choose',['/scratch/pk3251/appdir/Github/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/clapping/train/Clapping/2080.wav']),\
        # source_file=('choose',['/scratch/pk3251/appdir/Github/SpatialScaper/datasets/sound_event_datasets/FSD50K_FMA/telephone/test/Ringtone/16774.wav']),\
        source_time=('const',0.01),\
        event_time=("const", 0.05),\
        event_position=("const", [[event_position_x, event_position_y, event_position_z]]),\
        snr=("uniform", 10, 11))  # randomly choosing and spatializing an FSD50K sound event
        

    audiofile = os.path.join(OUTPUT_DIR, FORMAT, track_name)
    labelfile = os.path.join(OUTPUT_DIR, "labels", track_name)

    ssc.generate(audiofile, labelfile)


# Main loop for generating soundscapes
for iscape in range(NSCAPES):
    print(f"Generating soundscape: {iscape + 1}/{NSCAPES}")
    generate_soundscape(iscape)
