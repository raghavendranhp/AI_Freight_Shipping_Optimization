import pandas as pd
import random
from faker import Faker
import datetime

#Initialize Faker with Indian locale
fake = Faker('en_IN')

#Define our categorical variables
cities = ['Chennai', 'Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Kolkata', 'Pune', 'Ahmedabad', 'Coimbatore', 'Erode']
modes = ['Truck', 'Rail', 'Flight']
weathers = ['Clear', 'Rain', 'Fog', 'Storm']
traffics = ['Low', 'Medium', 'High']

#Configuration
num_records = 2000
data = []


route_distances = {}

def get_consistent_distance(origin, destination, mode):
    # frozenset makes order not matter: frozenset(['A', 'B']) == frozenset(['B', 'A'])
    route_key = (frozenset([origin, destination]), mode)
    
    # If we haven't calculated this exact route's distance yet, generate and save it
    if route_key not in route_distances:
        # Note: Expanded the distance ranges slightly to make Indian city distances realistic
        if mode == 'Truck':
            route_distances[route_key] = random.randint(100, 2500) 
        elif mode == 'Rail':
            route_distances[route_key] = random.randint(150, 2800)
        else: # Flight
            route_distances[route_key] = random.randint(300, 2500)
            
    #Return the locked-in distance
    return route_distances[route_key]

print("Generating synthetic freight data...")

for i in range(1, num_records + 1):
    #Generate basic categorical features
    shipment_id = f"S{i:04d}"
    origin = random.choice(cities)
    destination = random.choice([c for c in cities if c != origin]) # Ensure dest != origin
    mode = random.choice(modes)
    weather = random.choice(weathers)
    traffic = random.choice(traffics)
    
    #Generate a random departure time
    departure_time = fake.time_object().strftime("%H:%M")
    
    #Get the consistent distance for this city pair and mode
    distance = get_consistent_distance(origin, destination, mode)
        
    #Calculate Delay with realistic correlations
    # Base delay is somewhat random
    delay = random.randint(0, 20) 
    
    # Traffic impact
    if traffic == 'Medium':
        delay += random.randint(10, 30)
    elif traffic == 'High':
        delay += random.randint(30, 90)
        
    # Weather impact
    if weather == 'Rain':
        delay += random.randint(15, 40)
    elif weather == 'Fog':
        delay += random.randint(20, 60)
    elif weather == 'Storm':
        delay += random.randint(60, 150)
        
    #Long distances add slightly higher chance of random delays
    if distance > 1000:
        delay += random.randint(10, 45)
        
    #Append to data list
    data.append([
        shipment_id, origin, destination, distance, mode, 
        weather, traffic, departure_time, delay
    ])

#Create DataFrame
columns = [
    'ShipmentID', 'Origin', 'Destination', 'Distance', 
    'Mode', 'Weather', 'Traffic', 'DepartureTime', 'Delay'
]
df = pd.DataFrame(data, columns=columns)

#Save 2 CSV
csv_filename = 'freight_shipping_data.csv'
df.to_csv(csv_filename, index=False)

print(f"Success! Generated {num_records} records and saved to '{csv_filename}'.")
print("\nSample Data:")
print(df.head())