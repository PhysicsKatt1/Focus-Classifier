'''
##### SET A #####
##### problem 1 #####
#----- globals -----#
activity = [
    {"user": "Alice", "pages": 4},
    {"user": "Bob", "pages": 2},
    {"user": "Alice", "pages": 3},
    {"user": "Charlie", "pages": 5},
    {"user": "Bob", "pages": 4},
]

#----- calculate total pages read per person -----#
def analyze_activity(activity):
    counts = {}

    for page in activity:
        user = page['user']
        pages = page['pages']

        if user not in counts:
            counts[user] = 0
        
        counts[user] += pages

    return counts

print(analyze_activity(activity))

##### problem 2 #####
#----- globals -----#
events = [
    "login",
    "error",
    "login",
    "logout",
    "error",
    "login",
    "error",
]

#----- count error flag occurences -----#
def summarize_events(events):
    counts = {}

    for event in events:
        if event not in counts:
            counts[event] = 0
        
        counts[event] += 1

    return counts

print(summarize_events(events))

##### problem 3 #####
#----- globals -----#
reviews = [
    ("laptop", 5),
    ("phone", 4),
    ("laptop", 3),
    ("tablet", 5),
    ("phone", 5),
    ("laptop", 4),
]

#----- calculate average rating per device -----#
def average_ratings(reviews):
    ratings = {}
    devices = {}
    averages = {}

    for review in reviews:
        device = review[0]
        rating = review[1]

        if device not in ratings:
            ratings[device] = 0

        ratings[device] += rating 

        if device not in devices:
            devices[device] = 0

        devices[device] += 1
    
   
    for (entry, val), (entry1, val1) in zip(ratings.items(), devices.items()):
        averages[entry] = (val/val1)

    return averages

print(average_ratings(reviews))

##### SET B #####
##### problem 1 #####
#---gloabls---#
expenses = [
    ("Engineering", 120),
    ("Marketing", 75),
    ("Engineering", 50),
    ("Sales", 200),
    ("Marketing", 125),
    ("Engineering", 80),
    ("Sales", 50),
]

#---calucalte total expenses by department ---#
def total_expenses(expenses):
    total = {}

    for expense in expenses:
        department = expense[0]
        amount = expense[1]

        if department not in total:
            total[department] = 0
        
        total[department] += amount

    return total

print(total_expenses(expenses))

##### problem 2 #####
#---gloabls---#
sessions = [
    ("Alice", 25),
    ("Bob", 40),
    ("Alice", 15),
    ("Charlie", 30),
    ("Bob", 20),
    ("Alice", 10),
]

#---calculate total time per person ---#
def total_session_time(sessions):
    total = {}

    for session in sessions:
        name = session[0]
        time = session[1]

        if name not in total:
            total[name] = 0
        
        total[name] += time

    return total

print(total_session_time(sessions))

##### problem 3 #####
#---gloabls---#
temperatures = [
    72,
    85,
    68,
    91,
    77,
    95,
    70,
]

#--- return only temp > 80 ---#
def high_temperatures(temperatures):
    high = []

    for temp in temperatures:
        if temp > 80:
            high.append(temp)
    
    return high

print(high_temperatures(temperatures))
'''
##### SET C #####
##### problem 1 #####
#----- globals -----#
purchases = [
    ("Engineering", 120),
    ("Sales", 80),
    ("Engineering", 45),
    ("Marketing", 100),
    ("Sales", 60),
    ("Engineering", 35),
]

#--- calculate total spending per department ---#
def department_spending(purchases):
    total = {}

    for purchase in purchases:
        dept = purchase[0]
        amount = purchase[1]

        if dept not in total:
            total[dept] = 0
        
        total[dept] += amount

    return total

print(department_spending(purchases))

##### problem 2 #####
#----- globals -----#
user_ids = [
    "u101",
    "u205",
    "u101",
    "u310",
    "u205",
    "u450",
]

#--- list duplicates ---#
def find_duplicates(user_ids):
    duplicates = []
    dup = []

    for id in user_ids:
        if id not in duplicates:
            duplicates.append(id)
        
        else:
            dup.append(id)

    return dup

print(find_duplicates(user_ids))

##### problem 3 #####
#----- globals -----#
measurements = [
    5,
    42,
    101,
    75,
    9,
    100,
    10,
    150,
]

#--- save measurements between 10 and 100 ---#
def valid_measurements(measurements):
    valid = []

    for measurement in measurements:
        if measurement >= 10 and measurement <= 100:
            valid.append(measurement)

    return valid 

print(valid_measurements(measurements))

##### problem 4 #####
#----- globals -----#
transactions = [
    ("Alice", 50),
    ("Bob", 20),
    ("Alice", 30),
    ("Charlie", 10),
    ("Bob", 40),
    ("Alice", -20),
]

#--- calculate total balance per user ---#
def analyze_transactions(transactions):
    total = {}
    total_over50 = []

    for transaction in transactions:
        name = transaction[0]
        amount = transaction[1]

        if name not in total:
            total[name] = 0
        
        total[name] += amount
    
    for item in total.items():
        if item[1] >= 50:
            total_over50.append(item)
    
    total_over50 = dict(total_over50)

    return total_over50

print(analyze_transactions(transactions))