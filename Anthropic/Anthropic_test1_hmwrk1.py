##### SET A #####

##### problem 1 #####
all_sales = [
    {"store": "A", "sales": 120},
    {"store": "B", "sales": 85},
    {"store": "A", "sales": 75},
    {"store": "C", "sales": 200},
    {"store": "B", "sales": 40},
    {"store": "A", "sales": 55},
]

def total_sales_by_store(all_sales):
    totals = {}

    for sale in all_sales:
        store = sale['store']
        sales = sale['sales']

        if store not in totals:
            totals[store] = 0
        
        totals[store] += sales

    return totals

totals = total_sales_by_store(all_sales)
print(totals)

##### problem 2 #####
work = [
    {"project": "Alpha", "minutes": 45},
    {"project": "Beta", "minutes": 30},
    {"project": "Alpha", "minutes": 60},
    {"project": "Gamma", "minutes": 20},
    {"project": "Beta", "minutes": 50},
    {"project": "Alpha", "minutes": 15},
]

def total_minutes_by_project(work):
    totals = {}

    for w in work:
        project = w['project']
        minutes = w['minutes']

        if project not in totals:
            totals[project] = 0
        
        totals[project] += minutes

    return totals

totals = total_minutes_by_project(work)
print(totals)

##### problem 3 #####
meals = [
    {"food": "apple", "calories": 95},
    {"food": "banana", "calories": 105},
    {"food": "apple", "calories": 95},
    {"food": "chicken", "calories": 250},
    {"food": "banana", "calories": 105},
]

def total_calories_by_food(meals):
    totals = {}

    for meal in meals:
        food = meal['food']
        calories = meal['calories']

        if food not in totals:
            totals[food] = 0
        
        totals[food] += calories

    return totals

totals = total_calories_by_food(meals)
print(totals)

##### problem 4 #####
inventory = [
    {"product": "laptop", "units": 5},
    {"product": "phone", "units": 12},
    {"product": "laptop", "units": 3},
    {"product": "tablet", "units": 7},
    {"product": "phone", "units": 8},
    {"product": "tablet", "units": 2},
]

def total_inventory(inventory):
    totals = {}

    for item in inventory:
        product = item['product']
        units = item['units']

        if product not in totals:
            totals[product] = 0
        
        totals[product] += units

    return totals

totals = total_inventory(inventory)
print(totals)

##### problem 5 #####
donations = [
    {"organization": "Red", "amount": 50},
    {"organization": "Blue", "amount": 100},
    {"organization": "Red", "amount": 25},
    {"organization": "Green", "amount": 75},
    {"organization": "Blue", "amount": 50},
    {"organization": "Red", "amount": 25},
]

def total_donations(donations):
    totals = {}

    for donation in donations:
        organization = donation['organization']
        amount = donation['amount']

        if organization not in totals:
            totals[organization] = 0

        totals[organization] += amount
    return totals

totals = total_donations(donations)
print(totals)

##### SET B #####
##### problem 1 #####
#----- globals -----#
purchases = [
    "Alice:20",
    "Bob:15",
    "Alice:10",
    "Charlie:30",
    "Bob:5",
]

# purchases = []

#----- create dict from list data -----#
dicts = []
for item in purchases:
    try:
        name = item.split(':')[0]
        amount = item.split(':')[1]

        dicts.append({'Name': name, 'Amount': amount})
    except:
        continue

#----- caluclate total purchase amount -----#
total = {}

for purchase in dicts:
    name = purchase['Name']
    amount = int(purchase['Amount'])

    # print(name)

    if name not in total:
        total[name] = 0
    
    total[name] += amount

print(total)

##### problem 2 #####
#----- globals -----#
tasks = [
    ("Sarah", 4),
    ("Mike", 2),
    ("Sarah", 3),
    ("John", 5),
    ("Mike", 4),
    ("Sarah", 2),
]

#----- create dict from list data -----#
dicts = []
for item in tasks:
    try:
        name = item[0]
        task_num = item[1]

        dicts.append({'Name': name, 'Tasks': task_num})
    except:
        continue

#----- count total tasks per person -----#
total = {}

for item in dicts:
    name = item['Name']
    tasks = item['Tasks']

    if name not in total:
        total[name] = 0
    
    total[name] += int(tasks)

print(total)
    
##### problem 3 #####
#----- globals -----#
transactions = [
    ("checking", 100),
    ("savings", 500),
    ("checking", -25),
    ("checking", 50),
    ("savings", -100),
    ("checking", -10),
]

#----- create dict from data -----#
dicts = []

for transaction in transactions:
    account = transaction[0]
    amount = transaction[1]

    dicts.append({'Account': account, 'Amount': amount})

#----- calculate account balances -----#
balances = {}

for item in dicts:
    account = item['Account']
    amount = item['Amount']

    if account not in balances:
        balances[account] = 0
    
    balances[account] += amount
