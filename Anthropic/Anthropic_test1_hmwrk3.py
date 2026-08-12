'''
##### SET A #####

##### problem 1 #####
#--- globals ---#
numbers = [4, -2, 7, -5, 10, -1, 3]

#--- filter out negatives ---#
def positive_numbers(numbers):
    positives = []

    for num in numbers:
        if num >= 0:
            positives.append(num)

    return positives

print(positive_numbers(numbers))

##### problem 2 #####
#--- globals ---#
numbers = [12, 5, 19, 7, 3, 15]

#--- find the largest number ---#
def find_largest(numbers):
    largest = numbers[0]

    for num in numbers:
        if num > largest:
            largest = num
       
    return largest

print(find_largest(numbers))

##### problem 3 #####
#--- globals ---#
numbers = [2, 4, 6, 8]

#--- multiply each list value by 2 ---#
def double_numbers(numbers):
    double = []
    for num in numbers:
        double.append(2 * num)

    return double

print(double_numbers(numbers))

##### problem 4 #####
#--- globals ---#
numbers = [3, 8, 11, 14, 17, 20, 25]

#--- keep only even numbers ---#
def even_numbers(numbers):
    evens = []
    
    for num in numbers:
        if num/2 == int(num/2):
            evens.append(num)
    
    return evens

print(even_numbers(numbers))

##### problem 5 #####
#--- globals ---#
numbers = [10, 20, 30, 40, 50]

#--- keep only values greater than the list average ---#
def above_average(numbers):
    ave = sum(numbers) / len(numbers)
    average = []

    for num in numbers:
        if num > ave:
            average.append(num)

    return average

print(above_average(numbers))

##### SET B #####

##### problem 1 #####
#--- globals ---#
measurements = [12, 8, 15, 21, 7, 18, 10]

#--- keep acceptable measurements 10 - 18 inclusive ---#
def acceptable_measurements(measurements):
    acceptable = []
    
    for measurement in measurements:
        if measurement >= 10 and measurement <= 18:
            acceptable.append(measurement)
    
    return acceptable

print(acceptable_measurements(measurements))

##### problem 2 #####
#--- globals ---#
runs = [120, 85, 210, 175, 95, 240, 160]

#--- keep max images per run ---#
def best_run(runs):
    largest = runs[0]

    for run in runs:
        if run > largest:
            largest = run

    return largest

print(best_run(runs))

##### problem 3 #####
#--- globals ---#
scores = [10, 20, 30, 40]

#--- normalize by 10 ---#
def normalize_scores(scores):
    normalized = []
    norm_factor = 10

    for score in scores:
        normalized.append(score / norm_factor)

    return normalized

print(normalize_scores(scores))

##### problem 4 #####
#--- globals ---#
failures = [0, 3, 7, 1, 12, 0, 5, 2]
thresh = 5

#--- keep problematic builds ---#
def problematic_builds(failures):
    problems = []

    for failure in failures:
        if failure > thresh:
            problems.append(failure)

    return problems

print(problematic_builds(failures))

##### problem 5 #####
#--- globals ---#
measurements = [92, 105, 97, 110, 101, 89]
target = 100

#--- find the value closest to the target ---#
def closest_to_target(measurements, target):
    distance = abs(measurements[0] - target)
    value = measurements[0]

    for measurement in measurements:
        if abs(measurement - target) < distance:
            distance = abs(measurement - target)
            value = measurement

    return value

print(closest_to_target(measurements, target))

##### SET C #####

##### problem 1 #####
#--- globals ---#
names = [
    " Alice ",
    "Bob",
    " Charlie ",
    "Dana ",
]

#--- clean data ---#
def clean_names(names):
    cleaned = []
    for name in names:
        cleaned.append(name.strip(' '))

    return cleaned

print(clean_names(names))

##### problem 2 #####
#--- globals ---#
labels = [
    "FOCUS",
    "DEFECT",
    "IMAGE",
    "FOCUS",
    "ALIGNMENT",
]

#--- normalize labels ---#
def normalize_labels(labels):
    normalized = []

    for label in labels:
        normalized.append(label.lower())

    return normalized

print(normalize_labels(labels))

##### problem 3 #####
#--- globals ---#
emails = [
    "alice@gmail.com",
    "bob@yahoo.com",
    "charlie@company.com",
    "dana@gmail.com",
]

#--- extract username ---#
def extract_usernames(emails):
    username = []

    for email in emails:
        username.append(email.split('@')[0])

    return username

print(extract_usernames(emails))

##### problem 4 #####
#--- globals ---#
word = "banana"
character = 'n'

#--- count occurences of a letter in a string---#
def count_character(word, character):
    count = 0

    for letter in word:
        if letter == character:
            count += 1

    return count

print(count_character(word, character))

##### problem 5 #####
#--- globals ---#
labels = [
    "focus_error",
    "image_ok",
    "focus_warning",
    "alignment_error",
    "image_ok",
    "focus_error",
]

key = 'focus'

#--- find focus flags ---#
def find_focus_labels(labels, key):
    focus_only = []

    for label in labels:
        if key in label:
            focus_only.append(label)

    return focus_only

print(find_focus_labels(labels, key))


##### SET D #####

##### problem 1 #####
#--- globals ---#
image_ids = [
    "img01",
    "img02",
    "img01",
    "img03",
    "img02",
    "img04",
]

#--- find unique values ---#
def unique_images(image_ids):
    unique_ids = []

    for id in image_ids:
        if id not in unique_ids:
            unique_ids.append(id)
    
    return unique_ids

print(unique_images(image_ids))

##### problem 2 #####
#--- globals ---#
user_ids = [
    "u101",
    "u205",
    "u101",
    "u310",
    "u205",
    "u450",
]

#--- find duplicates ---#
def find_duplicates(user_ids):
    unique = []
    duplicates = []

    for id in user_ids:
        if id not in unique:
            unique.append(id)
        else:
            duplicates.append(id)

    return duplicates

print(find_duplicates(user_ids))

##### problem 3 #####
#--- globals ---#
system_a = ["A1", "A2", "A3", "A4"]
system_b = ["A2", "A4", "A5", "A6"]

#--- find common ids ---#
def common_ids(system_a, system_b):
    common = []

    for val in system_a:
        if val in system_b:
            common.append(val)

    return common

print(common_ids(system_a, system_b))

##### problem 4 #####
#--- globals ---#
sentence = "focus image focus alignment image focus"

#--- find the unique words ---#
def unique_words(sentence):
    unique = []

    sentence = sentence.split(' ')

    for word in sentence:
        if word not in unique:
            unique.append(word)

    return unique

print(unique_words(sentence))

##### problem 5 #####
#--- globals ---#
labels = [
    "focus",
    "defocus",
    "astigmatism",
    "focus",
    "alignment",
    "defocus",
]
# labels = ["focus", "focus", "focus"]

#--- find repeated labels ---#
def repeated_labels(labels):
    unique = set()
    repeated = []

    for label in labels:
        if label in unique:
            if label not in repeated:
                repeated.append(label)
        else:
            unique.add(label)
          
    return repeated

print(repeated_labels(labels))

'''

##### SET E #####

##### problem 1 #####
measurements = [42, 17, 31, 8, 25, 14]

def lowest_measurement(measurements):
    lowest = measurements[0]
    
    for measuremt in measurements:
        if measuremt < lowest:
            lowest = measuremt
    
    return lowest

print(lowest_measurement(measurements))

##### problem 2 #####
values = [95, 103, 108, 97, 112, 101]
target = 100

def closest_value(values, target):
    distance = abs(values[0] - target)
    closest = values[0]

    for value in values:
        if abs(value - target) < distance:
            distance = abs(value - target)
            closest = value

    return closest

print(closest_value(values, target))

##### problem 3 #####
machine_a = ["FIB", "SEM", "TEM", "AFM"]
machine_b = ["SEM", "AFM", "XRD", "Raman"]

def shared_devices(machine_a, machine_b):
    shared = []

    for val in machine_a:
        if val in machine_b:
            shared.append(val)

    return shared

print(shared_devices(machine_a, machine_b))

##### problem 4 #####
errors = [
    "E101",
    "E205",
    "E101",
    "E301",
    "E205",
    "E101",
    "E401",
]

def unique_errors(errors):
    unique = []

    for error in errors:
        if error in unique:
            continue
        else:
            unique.append(error)

    return unique

print(unique_errors(errors))

##### problem 5 #####
errors = [
    "E101",
    "E205",
    "E101",
    "E301",
    "E205",
    "E101",
    "E205",
]

def most_common_error(errors):
    unique = set()
    repeated = []

    for error in errors:
        if error in unique:
            if error not in repeated:
                repeated.append(error)
        else:
            unique.add(error)

    return repeated

print(most_common_error(errors))
