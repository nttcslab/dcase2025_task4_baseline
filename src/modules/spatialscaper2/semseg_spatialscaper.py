import sofa
import numpy as np
import json
import tqdm
from collections import namedtuple
import librosa
import scipy.fft
from scipy.signal import correlate
import random
import os
import warnings
from typing import Literal
import wave
import pysofaconventions as pysofa
import math
import glob

# original spatialscaper
from spatialscaper.core import Scaper
from spatialscaper.utils import (
    # get_files_list,
    # db2multiplier,
    traj_2_ir_idx,
    find_indices_of_change,
    IR_normalizer,
)
from spatialscaper.sofa_utils import load_rir_pos, load_pos
# from spatialscaper.spatialize import spatialize

def get_files_list_mod(path, split, extension, recursive=False): # extension including dot, e.g., '.sofa'. * for everything
    if split:
        subfiles = glob.glob(os.path.join(path, split, f"*{extension}"), recursive=recursive)
    else:
        subfiles = glob.glob(os.path.join(path, f"*{extension}"), recursive=recursive)
    return subfiles

def db2multiplier(target_db, x):
    '''
    return a scale so that scale*x has the power of target_db
    target_db
    x: signal
    '''
    energy_db = 10*np.log10(np.mean(np.square(x)))
    if not np.isfinite(energy_db):
        warnings.warn(f'All-zero signals was used')
        return 1
    else:
        return 10 ** ((target_db - energy_db) / 20)

def spatialize(
    audio,
    irs,
    **kwargs,
    ):
    '''
    Generate spatial audio from the dry audio and impulse responses

    return dry source (direct path and early reflection) at ref_channel if ref_channel and direct_path_time_sp are specified
    '''
    ref_channel = kwargs['ref_channel'] if 'ref_channel' in kwargs else -1
    direct_path_time_sp = kwargs['direct_path_time_sp'] if 'direct_path_time_sp' in kwargs else None

    n_ch, n_irs, n_ir_samples = irs.shape
    assert n_irs==1, 'no moving sound events'

    # simple cases
    wet_source = scipy.signal.fftconvolve(audio[:, None], irs[:, 0].T, mode="full", axes=0) # [wlen (audio length + ir length - 1), nch]

    output = {
        'wet_source': wet_source,
    }

    if ref_channel != -1:
        peak = np.argmax(irs[ref_channel, 0, :], axis=0) # find peak of IR
        ir_direct_path = irs[ref_channel, 0, :].copy()
        if peak + direct_path_time_sp[1] < ir_direct_path.shape[0]:
            ir_direct_path[peak + direct_path_time_sp[1]:] = 0
        if peak - direct_path_time_sp[0] > 0:
            ir_direct_path[:peak - direct_path_time_sp[0]] = 0
        dry = scipy.signal.fftconvolve(audio, ir_direct_path, mode="full", axes=0)
        output['dry_source'] = dry

    return output

Event = namedtuple(
    "Event",
    [
        "label",
        "source_file",
        "source_time",
        "event_time",
        "event_duration",
        "snr",
        "role",
        "pitch_shift",
        "time_stretch",
        "event_position",
    ],
)

class SemgSegScaper(Scaper):
    def __init__(
        self,
        **kwargs):

        # arguments of SemgSegScaper
        if 'ref_channel' not in kwargs: kwargs['ref_channel'] = 0

        if 'return_dry' not in kwargs:
            kwargs['return_dry'] = False
        else: assert 'spatialize_direct_path_time_ms' in kwargs, 'spatialize_direct_path_time_ms must be specified when return_dry = True'
        if 'spatialize_direct_path_time_ms' not in kwargs: kwargs['spatialize_direct_path_time_ms'] = None
        if 'return_wet' not in kwargs: kwargs['return_wet'] = False
        if 'return_ir' not in kwargs: kwargs['return_ir'] = False
        if 'return_background' not in kwargs: kwargs['return_background'] = False

        # initialize original Scaper
        scaper_arg_names = ['duration', 'foreground_dir', 'rir_dir', 'fmt', 'room', 'use_room_ambient_noise', 'background_dir', 'sr', 'DCASE_format', 'max_event_overlap', 'max_event_dur', 'ref_db', 'speed_limit', 'max_sample_attempts']

        scaper_args = {kw:a for kw, a in kwargs.items() if kw in scaper_arg_names}
        scaper_args['DCASE_format'] = False

        super().__init__(**scaper_args)
        delattr(self, 'room') # explicitly require setting room information
        # delete some unused params, make sure they are not used accidentally
        delattr(self, 'use_room_ambient_noise')
        delattr(self, 'format')
        delattr(self, 'DCASE_format')
        self.label_rate = 10 # this is not set if DCASE_format = False, but it is required for some functions

        self.config = kwargs

        if kwargs['spatialize_direct_path_time_ms'] is not None:
            self.spatialize_direct_path_time_sp =[ # convert to time millisecond to time sample
                int( kwargs['spatialize_direct_path_time_ms'][0]*kwargs['sr']/1000),
                int( kwargs['spatialize_direct_path_time_ms'][1]*kwargs['sr']/1000),
            ]
        else:
            self.spatialize_direct_path_time_sp = None

    def generate_metadata(self, metadata_path):
        '''
        Generate a json file, which can be use to reconstruct the same soundscape
        '''
        self.fg_events = sorted(self.fg_events, key=lambda x: x.event_time)
        data = {'config': self.config}
        data['fg_events'] = [event._asdict() for event in self.fg_events]
        data['bg_events'] = [event._asdict() for event in self.bg_events]
        data['room'] = self.room
        with open(metadata_path, "w") as outfile:
            json.dump(data, outfile, indent=4)

    def from_metadata(metadata_path,
                      return_dry=None,
                      return_wet=None,
                      return_ir=None,
                      return_background=None,):
        '''
        Create a SemgSegScaper object with same setting with the json file
        '''
        with open(metadata_path) as f:
            data = json.load(f);
        if 'config' not in data and 'init_args' in data:
            data['config'] = data['init_args'] # compatible with previous version
        obj = SemgSegScaper(**data['config'])
        obj.fg_events = [Event(**event_dict) for event_dict in data['fg_events']]
        obj.bg_events = [Event(**event_dict) for event_dict in data['bg_events']]
        obj.room = data['room']
        # overwrite if provided
        if return_dry is not None: obj.return_dry = return_dry
        if return_wet is not None: obj.return_wet = return_wet
        if return_ir is not None: obj.return_ir = return_ir
        if return_background is not None: obj.return_background = return_background

        sofa_path = os.path.join(obj.rir_dir, obj.room)
        sofa = pysofa.SOFAFile(sofa_path, "r")
        obj.nchans = sofa.getDimensionSize('R')
        sofa.close()

        return obj

    def get_room_list(self): # return full path
        return get_files_list_mod(self.rir_dir, None, '.sofa', recursive=True)
    def set_room(self, room_config=('choose', [])):
        '''
        Set the room (sofa file)
        The room should be placed in self.rir_dir
        TODO: change room to relative path to rir_dir
        '''
        if room_config[0] == 'const':
            room = room_config[1]; # basename
        if room_config[0] == 'choose':
            if room_config[1]:
                room = random.choice(room_config[1])
            else: # search and get random room
                # room_list = [fp for fp in get_files_list(self.rir_dir, None) if fp.endswith('.sofa')]
                room_list = self.get_room_list()
                if not room_list:
                    print(f'No room added: No sofa file in {self.rir_dir}')
                room = os.path.basename(random.choice(room_list))

        # validate room IR
        sofa_path = os.path.join(self.rir_dir, room)
        sofa = pysofa.SOFAFile(sofa_path, "r")
        if not sofa.isValid():
            print('Invalid sofa file, room not added')
        else:
            self.room = room
            # source_pos = sofa.getVariableValue("SourcePosition")
            self.nchans = sofa.getDimensionSize('R')
        sofa.close()

    def get_room_irs_xyz(self):
        # directly load from rir_dir/room
        room_sofa_path = os.path.join(
            self.rir_dir,
            self.room,
        )
        return load_pos(room_sofa_path, doas=False)

    def get_room_irs_wav_xyz(self, wav=True, pos=True):
        # directly load from rir_dir/room
        room_sofa_path = os.path.join( # new rooms
            self.rir_dir,
            self.room,
        )
        all_irs, ir_sr, all_ir_xyzs = load_rir_pos(room_sofa_path, doas=False)
        ir_sr = ir_sr.data[0]
        all_irs = all_irs.data
        all_ir_xyzs = all_ir_xyzs.data
        if ir_sr != self.sr:
            all_irs = librosa.resample(all_irs, orig_sr=ir_sr, target_sr=self.sr)
            ir_sr = self.sr
        return all_irs, ir_sr, all_ir_xyzs

    # def synthesize_events_and_labels(self, all_irs, all_ir_xyzs, out_audio):
    def synthesize_events_and_labels(self, out_audio):
        all_ir_xyzs = self.get_room_irs_xyz();
        room_sofa_path = os.path.join(self.rir_dir, self.room)
        sofafile = sofa.Database.open(room_sofa_path)

        # events = tqdm.tqdm(self.fg_events, desc="🧪 Spatializing events 🔊...")
        events = self.fg_events
        if self.config['return_wet']: wet_sources = []
        if self.config['return_dry']: dry_sources = []
        if self.config['return_ir']: irs_list = []
        for ievent, event in enumerate(events):
            # process relative path
            if event.source_file.startswith(self.foreground_dir):
                source_file_full = event.source_file
            else:
                source_file_full = os.path.join(self.foreground_dir, event.source_file)
            # fetch trajectory from irs
            ir_idx = traj_2_ir_idx(all_ir_xyzs, event.event_position)

            # irs = all_irs[ir_idx]
            irs = sofafile.Data.IR.get_values(indices = {'M':ir_idx})

            # load and normalize audio signal to have peak of 1
            # x, _ = librosa.load(event.source_file, sr=self.sr)
            x, _ = librosa.load(
                source_file_full,
                sr=self.sr,
                offset=event.source_time,
                duration=event.event_duration,
                mono=True,
            )
            x = x[: int(event.event_duration * self.sr)]

            if np.max(np.abs(x)) == 0:
                print(f"Warning: {source_file_full}")
            else:
                x = x / np.max(np.abs(x))
            # x = x / np.max(np.abs(x))
            # normalize irs to have unit energy
            norm_irs = IR_normalizer(irs)

            # no moving sound sources
            assert len(irs) == 1, 'No moving sound sources'
            # END modification <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            norm_irs = np.transpose(
                norm_irs, (1, 0, 2)
            )  # (n_irs, n_ch, n_ir_samples) -> (n_ch, n_irs, n_ir_samples)


            spatial_data = spatialize(
                x,
                norm_irs,
                ref_channel=self.config['ref_channel'],
                direct_path_time_sp=self.spatialize_direct_path_time_sp)
            xS = spatial_data['wet_source']

            # standardize the spatialized audio
            event_scale = db2multiplier(self.ref_db + event.snr, xS)
            xS = event_scale * xS

            # add to out_audio
            onsamp = int(event.event_time * self.sr)
            # output of spatialize has longer length than input
            max_length = min(onsamp + len(xS), out_audio.shape[0])
            out_audio[onsamp : max_length] += xS[: max_length - onsamp]

            if self.config['return_ir']: irs_list.append(norm_irs) # store irs for testing
            if self.config['return_wet']:
                wet_source = np.zeros_like(out_audio)
                wet_source[onsamp : max_length] = xS[: max_length - onsamp]
                wet_sources.append(wet_source)
            if self.config['return_dry']:
                dry_source = np.zeros(out_audio.shape[0], dtype=out_audio.dtype)
                dry_source[onsamp : max_length] = spatial_data['dry_source'][: max_length - onsamp]*event_scale # same scale as xS
                dry_sources.append(dry_source)

        return_obj = {
            'mixture': out_audio,
        }
        if self.config['return_dry']: return_obj['dry_sources'] = dry_sources
        if self.config['return_wet']: return_obj['wet_sources'] = wet_sources
        if self.config['return_ir']: return_obj['irs'] = irs_list
        # return_obj['original_sources'] = original_sources

        sofafile.close()

        return return_obj

    def generate(self):
        # initialize output audio array
        out_audio = np.zeros((int(self.duration * self.sr), self.nchans))

        # add background ambience
        out_audio = self.get_background_noise(out_audio)
        if self.config['return_background']: background_audio = out_audio.copy()

        # sort foreground events by onset time
        self.fg_events = sorted(self.fg_events, key=lambda x: x.event_time)

        # output = self.synthesize_events_and_labels(all_irs, all_ir_xyzs, out_audio)
        output = self.synthesize_events_and_labels(out_audio)

        output['labels'] = [e.label for e in self.fg_events]
        if self.config['return_background']: output['background'] = background_audio
        output['meta_data'] = self.fg_events
        return output

    def background_path_check(self, bg_path, room):
        '''
        Check if valid background file
        '''
        bg_fname = os.path.basename(bg_path)
        room_fname_noext = os.path.splitext(room)[0]
        return bg_fname.startswith(room_fname_noext)
    def add_background(self,
        source_file=("choose", []),
        split=None,
        ):

        label = None
        snr = ("const", 0)
        role = "background"
        pitch_shift = None
        time_stretch = None
        event_time = ("const", 0)
        event_duration = ("const", self.duration)
        event_position = None

        if source_file[0] == "choose" and source_file[1]:
            source_file_ = random.choice(source_file[1])
        elif source_file[0] == "choose":
            # bg_file_list = get_files_list(self.background_dir, split)
            bg_file_list = get_files_list_mod(self.background_dir, split, '.wav', recursive=True)
            bg_file_list = [fp for fp in bg_file_list if self.background_path_check(fp, self.room)]
            if not bg_file_list:
                warnings.warn(f'No valid background files (name starting with "{os.path.splitext(self.room)[0]}") found in "{self.background_dir}", No background added')
                return
            source_file_ = random.choice(bg_file_list)
        elif source_file[0] == "choose_wo_room_consistency":
            # bg_file_list = get_files_list(self.background_dir, split)
            bg_file_list = get_files_list_mod(self.background_dir, split, '.wav', recursive=True)
            if not bg_file_list:
                warnings.warn(f'No valid background files, No background added')
                return
            source_file_ = random.choice(bg_file_list)

        with wave.open(source_file_, 'rb') as f:
            # validate number of channels
            nchans = f.getnchannels()
            if nchans != self.nchans:
                print('invalid BG')
                warnings.warn(f'Background channel mismatch.\n{source_file_} has {nchans} channels, while impulse responses have {self.nchans} channels.\nNo background added.')
                return
            # calculate duration
            sr = f.getframerate()
            wlen = f.getnframes()
            ambient_noise_duration = wlen / float(sr)

        # ambient_noise_duration = librosa.get_duration(path=source_file_)
        if ambient_noise_duration > self.duration:
            # source_time = round(random.uniform(0, ambient_noise_duration - self.duration))
            source_time = math.floor(random.uniform(0, ambient_noise_duration - self.duration))
        else:
            source_time = None

        # remove the directory from source_file path
        if source_file_.startswith(self.background_dir):
            source_file_ = os.path.relpath(source_file_, start=self.background_dir)
        self.bg_events.append(
            Event(
                label=label,
                source_file=source_file_,
                source_time=source_time,
                event_time=event_time[1],
                event_duration=event_duration[1],
                event_position=event_position,
                snr=snr[1],
                role=role,
                pitch_shift=pitch_shift,
                time_stretch=time_stretch,
            )
        )

    def add_event(
        self,
        label=("choose", []),
        source_file=("choose", []),
        source_time=("const", 0),
        event_time=None,
        event_position=("choose", []),
        snr=("uniform", 5, 30),
        split=None,
    ):
        # TODO: pitch_shift=(pitch_dist, pitch_min, pitch_max),
        # TODO: time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))
        _DEFAULT_SNR_RANGE = (5, 30)
        if event_time is None:
            event_time = ("uniform", 0, self.duration)

        if label[0] == "choose" and label[1]:
            label_ = random.choice(label[1])
        elif label[0] == "choose":
            label_ = random.choice(list(self.fg_labels.keys()))
        elif label[0] == "const":
            label_ = label[1]
        elif label[0] == "choose_wo_replacement": # no duplicated sound events in a mixture
            added_labels = set(e.label for e in self.fg_events)
            remaining_labels = set(self.fg_labels.keys()) - added_labels
            label_ = random.choice(list(remaining_labels))

        if source_file[0] == "choose" and source_file[1]:
            source_file_ = random.choice(source_file[1])
        elif source_file[0] == "choose":
            # source_file_ = random.choice(
            #     get_files_list(os.path.join(self.foreground_dir, label_), split)
            # )
            source_file_ = random.choice(
                get_files_list_mod(os.path.join(self.foreground_dir, label_), split, '.wav', recursive=True)
            )


        event_duration_ = librosa.get_duration(path=source_file_)

        if source_time[0] == "const":
            source_time_ = source_time[1]
        if source_time[0] == "choose":
            if event_duration_ > self.max_event_dur:
                source_time_ = random.uniform(0, event_duration_ - self.max_event_dur)
            else:
                source_time_ = 0

        if event_duration_ > self.max_event_dur:
            event_duration_ = self.max_event_dur


        event_time_ = self.define_event_onset_time(
            event_time,
            event_duration_,
            self.fg_events,
            self.max_event_overlap,
            1 / self.label_rate,
        )
        if event_time_ is None:
            warnings.warn(
                f'Could not find a start time for sound event "{source_file_}" that satisfies max_event_overlap = {self.max_event_overlap}. If this continues happening, you may want to consider adding less sound events to the scape or increasing max_event_overlap.'
            )
            if source_file[0] == "choose":
                # TODO: why does this warning only print once?
                warnings.warn("Randomly choosing a new sound event to try again.")
                self.add_event(
                    label,
                    source_file,
                    source_time,
                    event_time,
                    event_position,
                    snr,
                    split,
                )
            return None

        # no moving sound sources
        # randomly select ir rather than selecting position
        if event_position[0] == "choose":
            if event_position[1]:
                event_position_ = [random.choice(event_position[1]).tolist()]
            else:
                all_positions = self.get_room_irs_xyz()
                rand_index = np.random.randint(0, all_positions.shape[0])
                event_position_ = [all_positions[rand_index].tolist()]
        elif event_position[0] == "const":
            event_position_ = [event_position[1].tolist()]

        if snr[0] == "choose" and snr[1]:
            snr_ = random.choice(snr[1])
        elif snr[0] == 'const':
            snr_ = snr[1]
        elif snr[0] == "uniform":
            assert len(snr) == 3, 'SNR range should be ["uniform", startSNR, endSNR ]'
            snr_ = random.uniform(*snr[1:])
        # else:
        #     snr_ = random.uniform(*_DEFAULT_SNR_RANGE)

        # remove the directory from source_file path
        if source_file_.startswith(self.foreground_dir):
            source_file_ = os.path.relpath(source_file_, start=self.foreground_dir)

        self.fg_events.append(
            Event(
                label=label_,
                source_file=source_file_,
                source_time=source_time_,
                event_time=event_time_,
                event_duration=event_duration_,
                event_position=event_position_,
                snr=snr_,
                role="foreground",
                pitch_shift=None,
                time_stretch=None,
            )
        )
    def get_background_noise(self, out_audio):
        for ievent, event in enumerate(self.bg_events):
            '''
            # source_file has been validated in add_background
            if not event.source_file:
                ambient = self.generate_noise(event)
            else:
            '''
            # process relative path
            if event.source_file.startswith(self.background_dir):
                source_file_full = event.source_file
            else:
                source_file_full = os.path.join(self.background_dir, event.source_file)

            if event.source_time is not None:
                ambient, _ = librosa.load(
                    # event.source_file,
                    source_file_full,
                    sr=self.sr,
                    offset=event.source_time,
                    duration=event.event_duration,
                    mono=False, # default True
                ) # [4, duration]
            else:  # repeat ambient file until scape duration
                # ambient, _ = librosa.load(event.source_file, sr=self.sr)
                ambient, _ = librosa.load(source_file_full, sr=self.sr, mono=False)
                total_samples = int(self.duration * self.sr)
                # repeats = -(-total_samples // len(ambient))  # ceiling division
                # ambient = np.tile(ambient, repeats)[:total_samples]
                repeats = -(-total_samples // ambient.shape[-1])
                ambient = np.repeat(ambient, repeats, axis=-1)[:, :total_samples]

            # ambient = ambient[:, np.newaxis]
            ambient = np.transpose(ambient, (1, 0))

            # scale = db2multiplier(self.ref_db + event.snr, np.mean(np.abs(ambient)))
            scale = db2multiplier(self.ref_db + event.snr, ambient) # modified db2multiplier
            out_audio += scale * ambient
        return out_audio
