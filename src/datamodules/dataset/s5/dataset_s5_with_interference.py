from .dataset_s5 import *

from src.modules.spatialscaper2.semseg_spatialscaper2 import SemgSegScaper2

class DatasetS5Interference(DatasetS5):
    def get_sound_scape(self, idx):
        if self.from_metadata: # generate sound scape from json config
            metadata_path = os.path.join(self.metadata_dir, self.data[idx]['metadata_path'])
            ssc = SemgSegScaper2.from_metadata(metadata_path)
        else: # randomly generate sound scape from param config
            # initialize object
            ssc = SemgSegScaper2(**self.spatialscaper)

            # set room
            ssc.set_room(('choose', [])) # random

            # add events
            nevents = random.randint(self.nevent_range[0], self.nevent_range[1])
            for i in range(nevents):
                ssc.add_event(
                    label=("choose_wo_replacement", []),
                    source_file=("choose", []),
                    source_time=("choose", []),
                    event_time=None,
                    event_position=('choose', []),
                    snr=("uniform", self.config['snr_range'][0], self.config['snr_range'][1]),
                    split=None,
                )
            assert self.nevent_range[0] <= len(ssc.fg_events) <=self.nevent_range[1]
            # import pdb; pdb.set_trace()
            if 'interference_dir' in self.config['spatialscaper']:
                ninteferences = random.randint(self.config['ninterference_range'][0], self.config['ninterference_range'][1])
                for _ in range(ninteferences):
                    ssc.add_interference(
                        label=("choose", []),
                        source_file=("choose", []),
                        source_time=("choose", []),
                        event_time=None,
                        event_position=('choose', []),
                        snr=("uniform", self.config['inteference_snr_range'][0], self.config['inteference_snr_range'][1]),
                        split=None,
                    )
            # add background, make sure it is consistent with room
            if self.spatialscaper['background_dir']: # only add noise if there is background_dir
                ssc.add_background(source_file = ('choose', []))
        output = ssc.generate()
        assert(len(set(output['labels'])) == len(output['labels'])), 'duplicated sound events in the mixture'
        return output
