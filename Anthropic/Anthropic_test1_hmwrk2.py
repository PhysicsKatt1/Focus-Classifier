##### SET A #####

##### problem 1 #####
#----- gloabls -----#
words = [
    "apple",
    "banana",
    "apple",
    "orange",
    "banana",
    "apple",
]

#----- count occurences of unique words -----#
def count_words(words):
    counts = {}

    for word in words:
        if word not in counts:
            counts[word] = 0
        
        counts[word] += 1

    return counts

counts = count_words(words)
print(counts)

##### problem 2 #####
#----- gloabls -----#
votes = [
    "Alice",
    "Bob",
    "Alice",
    "Charlie",
    "Bob",
    "Alice",
    "Charlie",
]

#----- count votes per person -----#
def count_votes(votes):
    counts = {}

    for vote in votes:
        if vote not in counts:
            counts[vote] = 0
        
        counts[vote] += 1

    return counts

counts = count_votes(votes)
print(counts)

##### problem 3 #####
#----- globals -----#
categories = [
    "A",
    "B",
    "A",
    "C",
    "B",
    "A",
    "C",
    "C",
]

#---- count number of occurences -----#
def count_categories(categories):
    counts = {}

    for category in categories:
        if category not in counts:
            counts[category] = 0
        
        counts[category] += 1

    return counts

counts = count_categories(categories)
print(counts)

##### problem 4 #####
#----- globals -----#
logins = [
    "Sarah",
    "Mike",
    "Sarah",
    "John",
    "Mike",
    "Sarah",
    "John",
    "Mike",
]

#---- count number of logins -----#
def login_counts(logins):
    counts = {}

    for login in logins:
        if login not in counts:
            counts[login] = 0

        counts[login] += 1

    return counts

counts = login_counts(logins)
print(counts)

##### problem 5 #####
#----- globals -----#
orders = [
    "laptop",
    "phone",
    "tablet",
    "laptop",
    "phone",
    "laptop",
    "phone",
    "phone",
]

#---- count number of ordered devices -----#
def order_counts(orders):
    counts = {}

    for order in orders:
        if order not in counts:
            counts[order] = 0

        counts[order] += 1

    return counts

counts = order_counts(orders)
print(counts)

##### SET B #####

##### problem 1 #####
#----- globals -----#
errors = [
    "E101",
    "E205",
    "E101",
    "E301",
    "E205",
    "E101",
    "E205",
]

#----- count frequency of each error flag -----#
def error_frequencies(errors):
    counts = {}

    for error in errors:
        if error not in counts:
            counts[error] = 0
        
        counts[error] += 1

    return counts

counts = error_frequencies(errors)
print(counts)

##### problem 2 #####
#----- globals -----#
scans = [
    "apple",
    "apple",
    "banana",
    "orange",
    "banana",
    "apple",
]

#----- count inventory -----#
def inventory_counts(scans):
    counts = {}

    for scan in scans:
        if scan not in counts:
            counts[scan] = 0
    
        counts[scan] += 1

    return counts

counts = inventory_counts(scans)
print(counts)

##### problem 3 #####
#----- globals -----#
tags = [
    ("image1", "cat"),
    ("image2", "dog"),
    ("image3", "cat"),
    ("image4", "bird"),
    ("image5", "dog"),
    ("image6", "cat"),
]

#----- count number of tags -----#
def tag_frequency(tags):
    counts = {}

    for tag in tags:
        if tag[1] not in counts:
            counts[tag[1]] = 0
        
        counts[tag[1]] += 1

    return counts

counts = tag_frequency(tags)
print(counts)