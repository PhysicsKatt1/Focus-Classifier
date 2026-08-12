'''
##### SET A #####
##### problem 1 #####
class SampleTracker:
    def __init__(self):
        self.samples = {}

        pass

    def add_sample(self, sample_id, material):
        self.samples[sample_id] = {'material': material, 'measurements': []}

        pass

    def get_material(self, sample_id):
        if sample_id in self.samples:
            return self.samples[sample_id]['material']
        else:
            return None

    def add_measurement(self, sample_id, value):
        if sample_id in self.samples:
            self.samples[sample_id]['measurements'].append(value)
            
            return True

        else:
            return False

    def get_measurements(self, sample_id):
        if sample_id in self.samples:
            # print(self.samples[sample_id]['measurements'])
            return self.samples[sample_id]['measurements']
        else:
            return None

tracker = SampleTracker()

tracker.add_sample("S001", "Silicon")
tracker.add_sample("S002", "Copper")

assert tracker.get_material("S001") == "Silicon"
assert tracker.get_material("S999") is None

assert tracker.add_measurement("S001", 12) is True
assert tracker.add_measurement("S001", 18) is True

assert tracker.get_measurements("S001") == [12, 18]
assert tracker.add_measurement("S999", 20) is False

##### problem 2 #####
class ImageLog:
    def __init__(self):
        self.images = {}

        pass

    def add_image(self, image_id, device, quality):
        self.images[image_id] = {'device': device, 'quality': quality}
        
        pass

    def get_image(self, image_id):
        if image_id in self.images:
            return self.images[image_id]
        

    def update_quality(self, image_id, quality):
        if image_id in self.images:
            self.images[image_id]['quality'] = quality
            
            return True
        
        else:
            return False


log = ImageLog()

log.add_image("IMG001", "SEM", 85)

assert log.get_image("IMG001") == {
    "device": "SEM",
    "quality": 85
}

assert log.update_quality("IMG001", 93) is True

assert log.get_image("IMG001") == {
    "device": "SEM",
    "quality": 93
}

assert log.update_quality("IMG999", 90) is False

##### problem 3 #####
class MeasurementLog:
    def __init__(self):
        self.runs = {}
        pass

    def add_run(self, run_id, device):
        self.runs[run_id] = {'device': device, 'measurement': []}
        
        pass

    def add_measurement(self, run_id, value):
        if run_id in self.runs:
            self.runs[run_id]['measurement'].append(value)
            print(self.runs)
            return True
        else:
            return False

    def summarize(self, run_id):
        summary = {}

        if run_id in self.runs:
            device = self.runs[run_id]['device']
            count = len(self.runs[run_id]['measurement'])

            summary['device'] = device
            summary['count'] = count

            print(summary)
            if count == 0:
                summary['minimum'] = None
                summary['maximum'] = None
            
            else:
                m = self.runs[run_id]['measurement'][0]
                l = self.runs[run_id]['measurement'][0]
                
                for measure in self.runs[run_id]['measurement']:
                    if measure > m:
                        m = measure
                    summary['maximum'] = m

                    if measure < l:
                        l = measure
                    summary['minimum'] = l
        
            return summary 
        
        else:
            return None
        

def test_measurement_log():

    log = MeasurementLog()

    # Add runs
    log.add_run("R001", "FIB")
    log.add_run("R002", "SEM")

    # # Empty run
    # assert log.summarize("R001") == {
    #     "device": "FIB",
    #     "count": 0,
    #     "minimum": None,
    #     "maximum": None
    # }

    # Add measurements
    assert log.add_measurement("R001", 15) is True
    assert log.add_measurement("R001", 8) is True
    assert log.add_measurement("R001", 21) is True

    # Unknown run
    assert log.add_measurement("R999", 50) is False

    # Summary
    assert log.summarize("R001") == {
        "device": "FIB",
        "count": 3,
        "minimum": 8,
        "maximum": 21
    }

    # Empty second run
    assert log.summarize("R002") == {
        "device": "SEM",
        "count": 0,
        "minimum": None,
        "maximum": None
    }

    # Unknown run
    assert log.summarize("R999") is None


test_measurement_log()
print("All tests passed!")

##### SET A #####
##### problem 1 #####
class MeasurementLog:
    def __init__(self):
        self.runs = {}

    def add_run(self, run_id, device):
        self.runs[run_id] = {'device': device, 'measurement': []}
        
        pass

    def add_measurement(self, run_id, value):
        if run_id in self.runs:
            self.runs[run_id]['measurement'].append(value)

            return True
        
        else:
            return False
        

    def count_above(self, run_id, threshold):
            if run_id in self.runs:
                counts = 0
                
                for measure in self.runs[run_id]['measurement']:
                    if measure > threshold:
                        counts += 1
                
                return counts
            
            else:
                return None           


def test_count_above():

    log = MeasurementLog()

    log.add_run("R001", "FIB")

    assert log.count_above("R001", 10) == 0

    log.add_measurement("R001", 5)
    log.add_measurement("R001", 15)
    log.add_measurement("R001", 20)
    log.add_measurement("R001", 8)

    assert log.count_above("R001", 10) == 2
    assert log.count_above("R001", 15) == 1
    assert log.count_above("R001", 20) == 0

    assert log.count_above("R999", 10) is None


test_count_above()
print("All tests passed!")

##### problem 2 #####
class RunLog:

    def __init__(self):
        self.runs = {}

    def add_run(self, run_id, device):
        self.runs[run_id] = {'device': device, 'measurements': []}
        
        pass

    def add_value(self, run_id, value):
        if run_id in self.runs:
            self.runs[run_id]['measurements'].append(value)

            return True
        else:
            return False

    def values_above(self, run_id, threshold):
        if run_id in self.runs:
            values = []

            for measure in self.runs[run_id]['measurements']:
                if measure > threshold:
                    values.append(measure)
        
            return values
        
        else:
            return None

def test_values_above():

    log = RunLog()

    log.add_run("R001", "FIB")

    assert log.values_above("R001", 10) == []

    assert log.add_value("R001", 5) is True
    assert log.add_value("R001", 15) is True
    assert log.add_value("R001", 20) is True
    assert log.add_value("R001", 8) is True
    assert log.add_value("R001", 25) is True

    assert log.values_above("R001", 10) == [15, 20, 25]
    assert log.values_above("R001", 20) == [25]
    assert log.values_above("R001", 25) == []

    assert log.values_above("R999", 10) is None


test_values_above()
print("All tests passed!")

##### problem 2 #####
class RunLog:
    def __init__(self):
        self.runs = {}

    def add_run(self, run_id, device):
        self.runs[run_id] = {'device': device, 'measurements': []}
        

    def add_value(self, run_id, value):
        if run_id in self.runs:
            self.runs[run_id]['measurements'].append(value)
            return True
        else:
            return None

    def average_value(self, run_id):
        if run_id in self.runs:
            if len(self.runs[run_id]['measurements']) != 0:
                return sum(self.runs[run_id]['measurements']) / len(self.runs[run_id]['measurements'])
            else: 
                return None

        else:
            return None

def test_average_value():

    log = RunLog()

    log.add_run("R001", "FIB")
    log.add_run("R002", "SEM")

    assert log.average_value("R001") is None

    log.add_value("R001", 10)
    log.add_value("R001", 20)
    log.add_value("R001", 30)

    assert log.average_value("R001") == 20

    log.add_value("R001", 40)

    assert log.average_value("R001") == 25

    assert log.average_value("R002") is None
    assert log.average_value("R999") is None


test_average_value()
print("All tests passed!")

##### problem 2 #####
class RunLog:
    def __init__(self):
        self.runs = {}

    def add_run(self, run_id, device):
        self.runs[run_id] = {'device':device, 'measurements':[]}

    def add_value(self, run_id, value):
        if run_id in self.runs:
            return self.runs[run_id]['measurements'].append(value)
        else: 
            return None

    def min_max(self, run_id):
        if run_id in self.runs:
            stats = {}
            if len(self.runs[run_id]['measurements']) == 0:
                stats['maximum'] = None
                stats['minimum'] = None

                return stats
            else:
                m = self.runs[run_id]['measurements'][0]
                l = self.runs[run_id]['measurements'][0]
                for measurement in self.runs[run_id]['measurements']:
                    if measurement > m:
                        m = measurement
                        stats['maximum'] = m
                    
                    if measurement < l:
                        l = measurement
                        stats['minimum'] = l
                
                return stats
                
        else:
          return None

    
def test_min_max():

    log = RunLog()

    log.add_run("R001", "FIB")
    log.add_run("R002", "SEM")

    assert log.min_max("R001") == {
        "minimum": None,
        "maximum": None
    }

    log.add_value("R001", 15)
    log.add_value("R001", 8)
    log.add_value("R001", 21)
    log.add_value("R001", 12)

    assert log.min_max("R001") == {
        "minimum": 8,
        "maximum": 21
    }

    assert log.min_max("R002") == {
        "minimum": None,
        "maximum": None
    }

    assert log.min_max("R999") is None


test_min_max()
print("All tests passed!")

'''

##### problem 3 #####
class RunLog:

    def __init__(self):
        self.runs = {}

    def add_run(self, run_id, device):
        self.runs[run_id] = {'device': device, 'measurements': []}
        
    def add_value(self, run_id, value):
        if run_id in self.runs:
            self.runs[run_id]['measurements'].append(value)
            return True
        else:
            return None

    def values_in_range(self, run_id, minimum, maximum):
        if run_id in self.runs:
            vals = []
            for measure in self.runs[run_id]['measurements']:
                if measure >= minimum and measure <= maximum:
                    vals.append(measure)
                
            return vals
                
        else:
            return None
              

def test_values_in_range():

    log = RunLog()

    log.add_run("R001", "FIB")

    log.add_value("R001", 5)
    log.add_value("R001", 10)
    log.add_value("R001", 15)
    log.add_value("R001", 20)
    log.add_value("R001", 25)

    assert log.values_in_range("R001", 10, 20) == [10, 15, 20]
    assert log.values_in_range("R001", 15, 25) == [15, 20, 25]
    assert log.values_in_range("R001", 11, 14) == []

    assert log.values_in_range("R999", 10, 20) is None


test_values_in_range()
print("All tests passed!")

