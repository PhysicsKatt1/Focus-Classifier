##### problem 1 #####
def count_values(values):
    counts = {}
    for value in values:
        if value not in counts:
            counts[value] = 1
        
        else:
             counts[value] += 1


    return counts

assert count_values([2, 3, 2, 5, 3, 2]) == {
    2: 3,
    3: 2,
    5: 1
}

assert count_values([]) == {}

assert count_values(["FIB", "SEM", "FIB"]) == {
    "FIB": 2,
    "SEM": 1
}

##### problem 2 #####
records = [
    ("FIB", 10),
    ("SEM", 20),
    ("FIB", 15),
    ("SEM", 25),
    ("TEM", 30),
]

def group_values(records):
    values = {}
    for record in records:
        device = record[0]
        val = record[1]

        if device not in values:
            values[device] = []
        
        values[device].append(val)

    return values

assert group_values([
    ("FIB", 10),
    ("SEM", 20),
    ("FIB", 15),
    ("SEM", 25),
    ("TEM", 30)
]) == {
    "FIB": [10, 15],
    "SEM": [20, 25],
    "TEM": [30]
}

assert group_values([]) == {}


##### problem 3 #####
devices = {
    "R001": "FIB",
    "R002": "SEM",
    "R003": "TEM",
}

def get_devices(run_ids):
    vals = []
    for id in run_ids:
        vals.append(devices.get(id))
            
    return vals
 

assert get_devices(["R003", "R001"]) == ["TEM", "FIB"]

assert get_devices(["R999"]) == [None]

assert get_devices(["R002", "R999", "R001"]) == [
    "SEM",
    None,
    "FIB"
]

assert get_devices([]) == []

##### problem 3 #####
measurements = [
    ("R001", 12),
    ("R002", 7),
    ("R003", 19),
    ("R004", 25),
]


def high_measurement_ids(measurements, threshold):
    val = []

    if len(measurements) == 0:
        return val
    
    else:
        for measurement in measurements:
            if measurement[1] > threshold:
                val.append(measurement[0])

        return val
             
assert high_measurement_ids(measurements, 10) == [
    "R001",
    "R003",
    "R004"
]

assert high_measurement_ids(measurements, 19) == [
    "R004"
]

assert high_measurement_ids(measurements, 25) == []

assert high_measurement_ids([], 10) == []

##### MINI TEST #####
class ExperimentLog:

    def __init__(self):
        self.experiment = {}

    def add_experiment(self, experiment_id, device):
        self.experiment[experiment_id] = {'device':device, 'measurements': []}

    def get_device(self, experiment_id):
        if experiment_id in self.experiment:
            return self.experiment[experiment_id]['device']
        else: 
            return None

    def add_measurement(self, experiment_id, value):
        if experiment_id in self.experiment:
            self.experiment[experiment_id]['measurements'].append(value)
            return True
        else:
            return False
        
    def get_measurements(self, experiment_id):
        if experiment_id in self.experiment:
            vals = []
            for measure in self.experiment[experiment_id]['measurements']:
                vals.append(measure)
            
            return vals
            
        else:
            return None

    def summarize(self, experiment_id):
        if experiment_id in self.experiment:
            summary = {}

            summary['device'] = self.experiment[experiment_id]['device']
            summary['count'] = len(self.experiment[experiment_id]['measurements'])  
            summary['minimum'] = []
            summary['maximum'] = []

            if summary['count'] == 0:
                summary['minimum'] = None
                summary['maximum'] = None
            else:
                m = self.experiment[experiment_id]['measurements'][0]
                l = self.experiment[experiment_id]['measurements'][0]

                for measurement in self.experiment[experiment_id]['measurements']:
                    if measurement > m:
                        m = measurement
                    if measurement < l:
                        l = measurement
                    
                    summary['minimum'] = l
                    summary['maximum'] = m

            return summary 
        else:
            return None

    def count_above(self, experiment_id, threshold):
        if experiment_id in self.experiment:
            counts = 0
            if len(self.experiment[experiment_id]['measurements']) == 0:
                return counts
            else:
                for measure in self.experiment[experiment_id]['measurements']:
                    if measure > threshold:
                        counts += 1
                return counts
    
        else:
            return None
        
    def values_in_range(self, experiment_id, minimum, maximum):
        if experiment_id in self.experiment:
            vals = []
            if len(self.experiment[experiment_id]['measurements']) == 0:
                return vals
            else:
                for measure in self.experiment[experiment_id]['measurements']:
                    if minimum <= measure and maximum >= measure:
                        vals.append(measure)
                return vals

        else:
            return None
            
        
def test_level_5():

    log = ExperimentLog()

    log.add_experiment("E001", "FIB")
    log.add_experiment("E002", "SEM")

    # Empty experiment
    assert log.values_in_range("E001", 10, 20) == []

    # Add measurements
    log.add_measurement("E001", 5)
    log.add_measurement("E001", 10)
    log.add_measurement("E001", 15)
    log.add_measurement("E001", 20)
    log.add_measurement("E001", 25)

    # Inclusive boundaries
    assert log.values_in_range("E001", 10, 20) == [10, 15, 20]

    # Narrower range
    assert log.values_in_range("E001", 11, 19) == [15]

    # No matches
    assert log.values_in_range("E001", 26, 30) == []

    # # Entire range
    assert log.values_in_range("E001", 5, 25) == [5, 10, 15, 20, 25]

    # Empty second experiment
    assert log.values_in_range("E002", 0, 100) == []

    # Unknown experiment
    assert log.values_in_range("E999", 10, 20) is None


test_level_5()