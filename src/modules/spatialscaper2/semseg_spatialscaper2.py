from .semseg_spatialscaper import *
from spatialscaper.utils import (
    get_label_list,
)

class SemgSegScaper2(SemgSegScaper):
    def __init__(
        self,
        **kwargs):

        self.int_events = []
        if 'interference_dir' in kwargs:
            int_label_list = get_label_list(kwargs['interference_dir'])
            self.int_labels = {l: i for i, l in enumerate(int_label_list)}
            if 'return_interference' not in kwargs: kwargs['return_interference'] = False

            self.interference_dir = kwargs['interference_dir']
        super().__init__(**kwargs)

    def generate_metadata(self, metadata_path):
        '''
        Generate a json file, which can be use to reconstruct the same soundscape
        '''
        self.fg_events = sorted(self.fg_events, key=lambda x: x.event_time)
        data = {'config': self.config}
        data['fg_events'] = [event._asdict() for event in self.fg_events]
        data['bg_events'] = [event._asdict() for event in self.bg_events]
        data['int_events'] = [event._asdict() for event in self.int_events]
        data['room'] = self.room
        with open(metadata_path, "w") as outfile:
            json.dump(data, outfile, indent=4)

    def from_metadata(metadata_path,
                      return_dry=None,
                      return_wet=None,
                      return_ir=None,
                      return_background=None,
                      return_interference=None):
        '''
        Create a SemgSegScaper object with same setting with the json file
        '''
        with open(metadata_path) as f:
            data = json.load(f);
        if 'config' not in data and 'init_args' in data:
            data['config'] = data['init_args'] # compatible with previous version
        obj = SemgSegScaper2(**data['config'])
        obj.fg_events = [Event(**event_dict) for event_dict in data['fg_events']]
        obj.bg_events = [Event(**event_dict) for event_dict in data['bg_events']]

        # interference events, compatible with previous version
        if 'interference_dir' in data['config']:
            obj.int_events = [Event(**event_dict) for event_dict in data['int_events']]
        else:
            obj.int_events = []

        obj.room = data['room']
        # overwrite if provided
        if return_dry is not None: obj.return_dry = return_dry
        if return_wet is not None: obj.return_wet = return_wet
        if return_ir is not None: obj.return_ir = return_ir
        if return_background is not None: obj.return_background = return_background
        if return_interference is not None: obj.return_interference = return_interference

        sofa_path = os.path.join(obj.rir_dir, obj.room)
        sofa = pysofa.SOFAFile(sofa_path, "r")
        obj.nchans = sofa.getDimensionSize('R')
        sofa.close()

        return obj


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
        self._add_event(
            target_event_list=self.fg_events,
            event_dir=self.foreground_dir,
            all_event_labels=self.fg_labels,
            label=label,
            source_file=source_file,
            source_time=source_time,
            event_time=event_time,
            event_position=event_position,
            snr=snr,
            split=split,
        )

    def add_interference(
        self,
        label=("choose", []),
        source_file=("choose", []),
        source_time=("const", 0),
        event_time=None,
        event_position=("choose", []),
        snr=("uniform", 5, 30),
        split=None,
    ):
        if 'interference_dir' not in self.config:
            warnings.warn(f'No interference_dir specified, No interference added')
            return

        self._add_event(
            target_event_list=self.int_events,
            event_dir=self.interference_dir,
            all_event_labels=self.int_labels,
            label=label,
            source_file=source_file,
            source_time=source_time,
            event_time=event_time,
            event_position=event_position,
            snr=snr,
            split=split,
        )

    def _add_event(
        self,
        target_event_list,
        event_dir,
        all_event_labels,

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
            label_ = random.choice(list(all_event_labels.keys()))
        elif label[0] == "const":
            label_ = label[1]
        elif label[0] == "choose_wo_replacement": # no duplicated sound events in a mixture
            added_labels = set(e.label for e in target_event_list)
            remaining_labels = set(all_event_labels.keys()) - added_labels
            label_ = random.choice(list(remaining_labels))

        if source_file[0] == "choose" and source_file[1]:
            source_file_ = random.choice(source_file[1])
        elif source_file[0] == "choose":
            # source_file_ = random.choice(
            #     get_files_list(os.path.join(event_dir, label_), split)
            # )
            source_file_ = random.choice(
                get_files_list_mod(os.path.join(event_dir, label_), split, '.wav', recursive=True)
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
            target_event_list,
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
                self._add_event(
                    target_event_list,
                    event_dir,
                    all_event_labels,

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
        if source_file_.startswith(event_dir):
            source_file_ = os.path.relpath(source_file_, start=event_dir)

        target_event_list.append(
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
            # x = x / np.max(np.abs(x))
            if np.max(np.abs(x)) == 0:
                print(f"Warning: {source_file_full}")
            else:
                x = x / np.max(np.abs(x))
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


        # add interference sound events
        if self.config['return_interference']: interference_sources = []
        for ievent, event in enumerate(self.int_events):
            # process relative path
            if event.source_file.startswith(self.interference_dir):
                source_file_full = event.source_file
            else:
                source_file_full = os.path.join(self.interference_dir, event.source_file)
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
            # x = x / np.max(np.abs(x))
            if np.max(np.abs(x)) == 0:
                print(f"Warning: {source_file_full}")
            else:
                x = x / np.max(np.abs(x))
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
                ref_channel=-1, # no need to return dry source
            )
            xS = spatial_data['wet_source']

            # standardize the spatialized audio
            event_scale = db2multiplier(self.ref_db + event.snr, xS)
            xS = event_scale * xS

            # add to out_audio
            onsamp = int(event.event_time * self.sr)
            # output of spatialize has longer length than input
            max_length = min(onsamp + len(xS), out_audio.shape[0])
            out_audio[onsamp : max_length] += xS[: max_length - onsamp]

            if self.config['return_interference']:
                wet_source = np.zeros_like(out_audio)
                wet_source[onsamp : max_length] = xS[: max_length - onsamp]
                interference_sources.append(wet_source)

        return_obj = {
            'mixture': out_audio,
        }
        if self.config['return_dry']: return_obj['dry_sources'] = dry_sources
        if self.config['return_wet']: return_obj['wet_sources'] = wet_sources
        if self.config['return_ir']: return_obj['irs'] = irs_list
        if self.config['return_interference']: return_obj['interference_sources'] = interference_sources
        # return_obj['original_sources'] = original_sources

        sofafile.close()

        return return_obj

    def generate(self):
        output = super().generate()
        if self.config['return_interference']: output['int_events'] = self.int_events
        if self.config['return_background']: output['bg_events'] = self.bg_events
        return output

