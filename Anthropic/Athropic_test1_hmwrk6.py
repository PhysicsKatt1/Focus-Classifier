'''
##### SET A #####
##### problem 1 #####
instruments = [
    "FIB",
    "SEM",
    "FIB",
    "TEM",
    "SEM",
    "FIB"
]

def count_instruments(instruments):
    counts = {}

    for val in instruments:
        if val not in counts:
            counts[val] = 0

        counts[val] += 1

    return counts

assert count_instruments(instruments) == {
    "FIB": 3,
    "SEM": 2,
    "TEM": 1
}

assert count_instruments([]) == {}

assert count_instruments(["FIB"]) == {
    "FIB": 1
}

##### problem 2 #####
measurements = {
    "S001": 12,
    "S002": 7,
    "S003": 21,
    "S004": 15
}

def samples_above(measurements, threshold):
    samples = []
    for item in measurements.items():
        if item[1] > threshold:
            # print(item[1])
            samples.append(item[0])
    print(samples)
    
    return samples

assert samples_above(measurements, 10) == [
    "S001",
    "S003",
    "S004"
]

assert samples_above(measurements, 21) == []

assert samples_above({}, 10) == []

##### problem 3 #####
records = [
    ("FIB", "E101"),
    ("SEM", "E205"),
    ("FIB", "E205"),
    ("FIB", "E101"),
    ("SEM", "E101"),
]

def group_errors(records):
    errors = {}
    if len(records) == 0:
        return errors
    else:
        for record in records:
            if record[0] not in errors:
                errors[record[0]] = [record[1]]
            
            else:
                errors[record[0]].append(record[1])

        return errors

assert group_errors(records) == {
    "FIB": ["E101", "E205", "E101"],
    "SEM": ["E205", "E101"]
}

assert group_errors([]) == {}

assert group_errors([
    ("FIB", "E101")
]) == {
    "FIB": ["E101"]
}

##### problem 4 #####

class ImageDatabase:

    def __init__(self):
        self.images = {}

    def add_image(self, image_id, device, score):
        self.images[image_id] = {'device': device, 'score': score}

    def get_image(self, image_id):
        if image_id in self.images:

            # print(self.images.get(image_id))

            return self.images.get(image_id)
        else:
            return None

    def update_score(self, image_id, score):
        if image_id in self.images:
            self.images[image_id]['score'] = score
            return True

        else:
            return False

    def scores_above(self, threshold):
        above = []
        for entry in self.images.items():
            if entry[1]['score'] > threshold:
                above.append(entry[0])

        return above

db = ImageDatabase()

db.add_image("IMG001", "SEM", 85)
db.add_image("IMG002", "FIB", 72)

assert db.get_image("IMG001") == {
    "device": "SEM",
    "score": 85
}

assert db.get_image("IMG002") == {
    "device": "FIB",
    "score": 72
}

assert db.get_image("IMG999") is None

assert db.update_score("IMG001", 93) is True

assert db.get_image("IMG001") == {
    "device": "SEM",
    "score": 93
}

assert db.update_score("IMG999", 90) is False

assert db.scores_above(80) == ["IMG001"]
assert db.scores_above(93) == []
assert db.scores_above(50) == ["IMG001", "IMG002"]

##### SET B #####

##### problem 1 #####
images = [
    ("IMG001", 82),
    ("IMG002", 67),
    ("IMG003", 91),
    ("IMG004", 75),
]

def high_quality_images(images, threshold):
    vals = []
    for image in images:
        if image[1] >= threshold:
            vals.append(image[0])
    return vals

assert high_quality_images(images, 80) == [
    "IMG001",
    "IMG003"
]

assert high_quality_images(images, 91) == [
    "IMG003"
]

assert high_quality_images(images, 92) == []

assert high_quality_images([], 80) == []

##### problem 2 #####
measurements = {
    "FIB": [10, 20, 30],
    "SEM": [5, 15],
    "TEM": [8, 12, 20]
}

def total_measurements(measurements):
    totals = {}

    for measurement in measurements.items():
        totals[measurement[0]] = sum(measurement[1])


    return totals
            

assert total_measurements(measurements) == {
    "FIB": 60,
    "SEM": 20,
    "TEM": 40
}

assert total_measurements({}) == {}

assert total_measurements({
    "FIB": []
}) == {
    "FIB": 0
}

assert total_measurements({
    "SEM": [10]
}) == {
    "SEM": 10
}


##### problem 3 #####
images = [
    ("IMG001", "SEM", 85),
    ("IMG002", "FIB", 72),
    ("IMG003", "SEM", 91),
    ("IMG004", "FIB", 88),
    ("IMG005", "SEM", 76),
]

def best_image_by_device(images):
    best = {}
    final = {}

    for image in images:
        img = image[0]
        device = image[1]
        val = image[2]

        if device not in best:
            best[device] = (img, val)

        elif val > best[device][1]:
            best[device] = (img, val)

        for device, entry in best.items():
            final[device] = entry[0]
        
    return final

assert best_image_by_device(images) == {
    "SEM": "IMG003",
    "FIB": "IMG004"
}

assert best_image_by_device([]) == {}

assert best_image_by_device([
    ("IMG001", "SEM", 80)
]) == {
    "SEM": "IMG001"
}

assert best_image_by_device([
    ("IMG001", "FIB", 80),
    ("IMG002", "FIB", 90),
    ("IMG003", "FIB", 85)
]) == {
    "FIB": "IMG002"
}


##### problem 4 #####
measurements = [
    ("M001", "FIB", 14),
    ("M002", "SEM", 22),
    ("M003", "FIB", 31),
    ("M004", "SEM", 18),
    ("M005", "FIB", 27),
]

def highest_measurement_by_device(measurements):
    highest = {}
    final = {}
    for measurement in measurements:
        m = measurement[0]
        device = measurement[1]
        val = measurement[2]

        if device not in highest:
            highest[device] = (m, val)

        elif val >  highest[device][1]:
            highest[device] = (m, val)

        for i, z in highest.items():
            final[i] = z[0]


    return final

assert highest_measurement_by_device(measurements) == {
    "FIB": "M003",
    "SEM": "M002"
}

assert highest_measurement_by_device([]) == {}

assert highest_measurement_by_device([
    ("M001", "SEM", 50)
]) == {
    "SEM": "M001"
}

assert highest_measurement_by_device([
    ("M001", "FIB", 10),
    ("M002", "FIB", 20),
    ("M003", "FIB", 15)
]) == {
    "FIB": "M002"
}

##### problem 6 #####
images = [
    ("IMG001", 80),
    ("IMG002", 60),
    ("IMG003", 100),
    ("IMG004", 70),
]

def below_average_images(images):
    below = []
    vals = []
    for image in images:
       vals.append(image[1])

    if len(vals) == 0:
        return below
    else:
        average = sum(vals) / len(vals)


    for image in images:
        if image[1] < average:
            print(image[0])
            below.append(image[0])
    return below

assert below_average_images(images) == [
    "IMG002",
    "IMG004"
]

assert below_average_images([]) == []

assert below_average_images([
    ("IMG001", 50)
]) == []

assert below_average_images([
    ("IMG001", 10),
    ("IMG002", 20),
    ("IMG003", 30)
]) == ["IMG001"]


#####
class ImageTracker:
    def __init__(self):
        self.images = {}

    def add_image(self, image_id, device, score):
        self.images[image_id] = {'device': device, 'score': score}

    def get_image(self, image_id):
        if image_id in self.images:
            return self.images.get(image_id)

    def update_score(self, image_id, score):
        if image_id in self.images:
            self.images[image_id]['score'] = score

            return True
        else:
            return False

    def scores_above(self, threshold):
        above = []
        for item, vals in self.images.items():
            device = vals['device']
            score = vals['score']
            img = item

            if score > threshold:
                above.append(img)

        return above

    def best_image_by_device(self):
        test = {}
        best = {}
        for item, vals in self.images.items():
            device = vals['device']
            score = vals['score']
            img = item

            if device not in test:
                test[device] = (img, score)
            elif score > test[device][1]:
                test[device] = (img, score)

        for item in test.items():
            if item[0] not in best:
                best[item[0]] = item[1][0]

        return best

def test_level_3():

    tracker = ImageTracker()

    tracker.add_image("IMG001", "SEM", 85)
    tracker.add_image("IMG002", "FIB", 72)
    tracker.add_image("IMG003", "SEM", 91)
    tracker.add_image("IMG004", "FIB", 88)

    assert tracker.best_image_by_device() == {
        "SEM": "IMG003",
        "FIB": "IMG004"
    }

    empty_tracker = ImageTracker()

    assert empty_tracker.best_image_by_device() == {}

test_level_3()


##### problem 1 #####
images = [
    ("IMG001", "SEM", 85),
    ("IMG002", "FIB", 72),
    ("IMG003", "SEM", 91),
    ("IMG004", "FIB", 88),
    ("IMG005", "SEM", 70),
]

def lowest_image_by_device(images):
    test = {}
    lowest = {}
    for image in images:
        img = image[0]
        device = image[1]
        val = image[2]

        if device not in test:
            test[device] = (img, val)

        elif val < test[device][1]:
            test[device] = (img, val)

    for item in test.items():
        if item[0] not in lowest:
            lowest[item[0]] = item[1][0]

    return lowest

assert lowest_image_by_device(images) == {
    "SEM": "IMG005",
    "FIB": "IMG002"
}

assert lowest_image_by_device([]) == {}

assert lowest_image_by_device([
    ("IMG001", "SEM", 50)
]) == {
    "SEM": "IMG001"
}


##### problem 2 #####
measurements = [
    ("M001", "FIB", 12),
    ("M002", "SEM", 20),
    ("M003", "FIB", 18),
    ("M004", "SEM", 15),
    ("M005", "FIB", 25),
]

def highest_measurement_by_device(measurements):
    test = {}
    highest = {}
    for measure in measurements:
        m = measure[0]
        device = measure[1]
        val = measure[2]

        if device not in test:
            test[device] = (m, val)
        elif val > test[device][1]:
            test[device] = (m, val)

    for item in test.items():
        if item[0] not in highest:
            highest[item[0]] = item[1][0]

    return highest

assert highest_measurement_by_device(measurements) == {
    "FIB": "M005",
    "SEM": "M002"
}

assert highest_measurement_by_device([]) == {}

assert highest_measurement_by_device([
    ("M001", "FIB", 10)
]) == {
    "FIB": "M001"
}

'''
##### problem 3 #####
images = [
    ("IMG001", "SEM", 85),
    ("IMG002", "FIB", 72),
    ("IMG003", "SEM", 91),
    ("IMG004", "FIB", 88),
]

def best_score_by_device(images):
    test = {}
    best = {}
    for image in images:
        img = image[0]
        device = image[1]
        score = image[2]

        if device not in test:
            test[device] = (img, score)
        elif score > test[device][1]:
            test[device] = (img, score)

    for item in test.items():
        print(item)
        if item[0] not in best:
            best[item[0]] = item[1][1]


    return best

assert best_score_by_device(images) == {
    "SEM": 91,
    "FIB": 88
}

assert best_score_by_device([]) == {}

assert best_score_by_device([
    ("IMG001", "SEM", 50)
]) == {
    "SEM": 50
}