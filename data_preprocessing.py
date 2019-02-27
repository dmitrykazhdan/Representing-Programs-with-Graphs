



class SampleMetaInformation():

    def __init__(self, sample_fname, node_id):
        self.fname = sample_fname
        self.node_id = node_id


    def get_var_type(self):
        return None

    def get_var_occurences(self):
        return None



class CorpusMetaInformation():
    def __init__(self):
        self.sample_meta_inf = []


    def add_sample_inf(self, sample_inf):
        self.sample_meta_inf.append(sample_inf)


    def process_sample_inf(self):
        return None










