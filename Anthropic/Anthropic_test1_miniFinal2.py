'''
##### Test 1 #####

##### problem 1 #####
readings = [14, 7, 22, 31, 9, 18, 25]

def acceptable_readings(readings):
    acceptable  = []

    for reading in readings:
        if reading >= 10 and reading < 25:
            acceptable.append(reading)

    return acceptable 

print(acceptable_readings(readings))

##### problem 2 #####
records = [
    "IMG001:focus",
    "IMG002:alignment",
    "IMG003:focus",
    "IMG004:defocus",
]

def focus_images(records):
    focus = []

    for record in records:
        split = record.split(':')
        if 'focus' in split[1]:   
            if len(split[1]) == len('focus'):        
                focus.append(split[0])

    return focus

print(focus_images(records))

##### problem 3 #####
scores = [71, 84, 63, 92, 77, 88]

def highest_score(scores):
    highest = scores[0]

    for score in scores:
        if score > highest:
            highest = score

    return highest

print(highest_score(scores))

##### problem 4 #####
samples = [
    "sample_A",
    "sample_B",
    "sample_A",
    "sample_C",
    "sample_B",
    "sample_D",
]

def repeated_samples(samples):
    repeated = []
    unique = set()

    for sample in samples:
        if sample in unique:
            if sample not in repeated:
                repeated.append(sample)
        else:
            unique.add(sample)
        

    return repeated

print(repeated_samples(samples))

##### problem 5 #####
measurements = [103, 97, 108, 94, 101, 110]
target = 100

def closest_measurement(measurements, target):
    distance = abs(measurements[0] - target)
    closest = measurements[0]

    for measurement in measurements:
        if abs(measurement - target) < distance:
            distance = abs(measurement - target)
            closest = measurement

    return closest

print(closest_measurement(measurements, target))

'''
##### Test 2 #####
class ExperimentLog:
    def __init__(self):
        self.runs = {}
        
        pass

    def add_run(self, run_id, device):
        self.runs[run_id] = {'device': device, 'meausurement': []}
        # print(self.runs)
        
        pass

    def get_device(self, run_id):
        if run_id in self.runs:
            # print(self.runs[run_id]['device'])
            return self.runs[run_id]['device']
        else:
            # print(None)
            return None
        
    def add_measurement(self, run_id, value):
        if run_id in self.runs:
            self.runs[run_id]['meausurement'].append(value)
            # print(self.runs)
            return True
        else:
            return False

    def get_measurements(self, run_id):
        if run_id in self.runs:
            # print(self.runs[run_id]['meausurement'])
            return self.runs[run_id]['meausurement']
        else:
            return None
        
    def summarize_run(self, run_id):
        summary = {}
        if run_id in self.runs:
            device = self.runs[run_id]['device']
            count = len(self.runs[run_id]['meausurement'])
            
            summary['device'] = device
            summary['count'] = count

            if summary['count'] == 0:
                summary['minimum'] = None
                summary['maximum'] = None

            else:
                m = self.runs[run_id]['meausurement'][0]
                l = self.runs[run_id]['meausurement'][0]

                for measure in self.runs[run_id]['meausurement']:
                    if measure > m:
                        m = measure
                    summary['maximum'] = m

                    if measure < l:
                        l = measure
                    summary['minimum'] = l
            return summary 
        
        else:
            return None

                   

        

def test_level_1():
    log = ExperimentLog()

    log.add_run("R001", "FIB")
    log.add_run("R002", "SEM")

    assert log.get_device("R001") == "FIB"
    assert log.get_device("R002") == "SEM"
    assert log.get_device("R999") is None

test_level_1()

def test_level_2():
    log = ExperimentLog()

    log.add_run("R001", "FIB")

    assert log.add_measurement("R001", 12) is True
    assert log.add_measurement("R001", 18) is True
    assert log.add_measurement("R001", 9) is True

    assert log.get_measurements("R001") == [12, 18, 9]

    assert log.add_measurement("R999", 20) is False
    assert log.get_measurements("R999") is None

test_level_2()

def test_level_3():
    log = ExperimentLog()

    log.add_run("R001", "FIB")

    assert log.summarize_run("R001") == {
        "device": "FIB",
        "count": 0,
        "minimum": None,
        "maximum": None
    }

    log.add_measurement("R001", 15)
    log.add_measurement("R001", 8)
    log.add_measurement("R001", 21)

    assert log.summarize_run("R001") == {
        "device": "FIB",
        "count": 3,
        "minimum": 8,
        "maximum": 21
    }

    assert log.summarize_run("R999") is None

test_level_3()