from flask import Flask, render_template, request, send_file
import pandas as pd
import folium
import networkx as nx
import random
import math
import requests
import time

app = Flask(__name__)

# Load pub data
df = pd.read_csv('open_pubs.csv')

# Clean data
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df = df.dropna(subset=['latitude', 'longitude'])
df = df.drop_duplicates(subset=['name'], keep='last')

# OSRM function to get the shortest route and distance between two locations
def get_osrm_route(start_coords, end_coords):
    url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&geometries=geojson"
    response = requests.get(url)
    
    if response.status_code == 200:
        route_data = response.json()
        distance_meters = route_data['routes'][0]['distance']  # Distance in meters
        route_coords = route_data['routes'][0]['geometry']['coordinates']
        return route_coords, distance_meters
    else:
        return None, None

# Function to calculate route distance
def calculate_route_distance(route, graph):
    total_distance = 0
    for i in range(len(route) - 1):
        total_distance += graph[route[i]][route[i + 1]]['weight']
    return total_distance

# Simulated Annealing algorithm to optimize the pub route
def simulated_annealing(graph, initial_route):
    current_route = initial_route.copy()
    current_distance = calculate_route_distance(current_route, graph)
    best_route = current_route
    best_distance = current_distance

    temperature = 1000.0
    cooling_rate = 0.9999
    while temperature > 1:
        new_route = current_route.copy()
        i, j = random.sample(range(len(new_route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        
        new_distance = calculate_route_distance(new_route, graph)
        if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temperature):
            current_route = new_route
            current_distance = new_distance
            
            if current_distance < best_distance:
                best_route = current_route
                best_distance = current_distance

        temperature *= cooling_rate

    return best_route, best_distance

def create_initial_population(population_size, pub_names):
    population = []
    for _ in range(population_size):
        route = pub_names[:]
        random.shuffle(route)
        population.append(route)
    return population

# Select the best individuals for mating (tournament selection)
def selection(population, graph, tournament_size=5):
    selected = []
    for _ in range(tournament_size):
        individual = random.choice(population)
        selected.append(individual)
    selected.sort(key=lambda x: calculate_route_distance(x, graph))
    return selected[0]  # Return the best individual in the tournament

# Crossover function to create a child route
def crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    # Inherit a subset from parent1
    child[start:end] = parent1[start:end]
    # Fill remaining positions with genes from parent2 in order
    ptr = 0
    for gene in parent2:
        if gene not in child:
            while child[ptr] is not None:
                ptr += 1
            child[ptr] = gene
    return child

# Mutation function to introduce some variations
def mutate(route, mutation_rate=0.01):
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route) - 1)
            route[i], route[j] = route[j], route[i]  # Swap two pubs
    return route

# Main Genetic Algorithm function
def genetic_algorithm(graph, pub_names, population_size=100, generations=500, mutation_rate=0.01):
    population = create_initial_population(population_size, pub_names)
    best_route = min(population, key=lambda x: calculate_route_distance(x, graph))
    best_distance = calculate_route_distance(best_route, graph)

    for generation in range(generations):
        new_population = []

        # Create new population through selection, crossover, and mutation
        for _ in range(population_size):
            parent1 = selection(population, graph)
            parent2 = selection(population, graph)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)

        # Update population and track the best route
        population = new_population
        current_best_route = min(population, key=lambda x: calculate_route_distance(x, graph))
        current_best_distance = calculate_route_distance(current_best_route, graph)

        if current_best_distance < best_distance:
            best_route = current_best_route
            best_distance = current_best_distance

        # Optionally, print progress every 50 generations
        # if generation % 50 == 0:
        #     print(f"Generation {generation}: Best Distance = {best_distance:.2f} km")

    return best_route, best_distance

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def bfs(graph, start):
    queue = []

    queue.append((start, {key: False for (key) in graph.nodes()} , 0, [start]))
    visited = {}

    while queue:
        current_city, visited_cities, current_cost, path = queue.pop(0)

        if all(val == True for val in visited_cities.values()):
            return path

        if (hashabledict(visited_cities) in visited and visited[visited_cities] <= current_cost):
            continue

        visited[hashabledict(visited_cities)] = current_cost        

        for next_city in list(graph[current_city]):
            if visited_cities[next_city] == False:
                visited_cities[next_city] = True
                current_cost += graph[current_city][next_city].get('weight', None)
                #print(current_cost)
                path.append(next_city)
                queue.append((next_city, visited_cities, current_cost, path))





@app.route('/')
def index():
    cities = df['local_authority'].unique()
    return render_template('index.html', cities=cities)

@app.route('/get_pubs/<city>', methods=['GET'])
def get_pubs(city):
    # Fetch pubs for the selected city
    pubs_in_city = df[df['local_authority'] == city]['name'].tolist()

    # Return the list of pubs as a JSON response
    return {'pubs': pubs_in_city}


@app.route('/optimize_route', methods=['POST'])
def optimize_route():
    chosen_city = request.form['city']

    algorithm_choice = request.form['algorithm']
    selected_pubs = request.form.getlist('selected_pubs')  # Get the list of selected pubs
    print(f"Selected pubs: {selected_pubs}")  # Print the selected pubs
    num_pubs = len(selected_pubs)
    print(f"num_pubs: {num_pubs}") 
    # Get all pubs for the chosen city and print them
    df_chosen_city = df[df['local_authority'] == chosen_city]
    print(f"All pubs in {chosen_city}: {df_chosen_city['name'].tolist()}")  # Check all pubs for the city
    
    # Now filter based on selected pubs
    df_chosen_city = df_chosen_city.head(num_pubs)
    df_chosen_city = df_chosen_city[df_chosen_city['name'].isin(selected_pubs)]
    
    print(f"Filtered pubs for {chosen_city}: {df_chosen_city.shape[0]} pubs")
    
    # Check what pubs are left after filtering
    print(f"Pubs remaining after filtering: {df_chosen_city['name'].tolist()}")

    pub_locations = list(zip(df_chosen_city['latitude'], df_chosen_city['longitude']))
    pub_names = df_chosen_city['name'].tolist()

    G = nx.Graph()

    for i, pub_name in enumerate(pub_names):
        G.add_node(pub_name, pos=pub_locations[i])

    for i in range(len(pub_locations) - 1):
        for j in range(i + 1, len(pub_locations)):  
            start_coords = pub_locations[i]
            end_coords = pub_locations[j]
            
            route, distance_meters = get_osrm_route(start_coords, end_coords)
            
            if route:
                distance_km = distance_meters / 1000  
                G.add_edge(pub_names[i], pub_names[j], weight=distance_km)

    initial_route = pub_names

    
    if algorithm_choice == 'bfs':
        optimized_route = bfs(G, pub_names[0]) 
        optimized_distance = calculate_route_distance(optimized_route, G)
    elif algorithm_choice == 'genetic_algorithm':
        optimized_route, optimized_distance = genetic_algorithm(G, pub_names)
    elif algorithm_choice == 'simulated_annealing':
        optimized_route, optimized_distance = simulated_annealing(G, initial_route)

    optimized_route.append(optimized_route[0])  

    m_optimized = folium.Map(location=[df_chosen_city['latitude'].mean(), df_chosen_city['longitude'].mean()], zoom_start=13)
    
    for index, row in df_chosen_city.iterrows():
        folium.Marker(location=[row['latitude'], row['longitude']],
                      popup=f"{row['name']}<br>{row['address']}<br>{row['postcode']}").add_to(m_optimized)

    for i in range(len(optimized_route) - 1):
        start_pub = optimized_route[i]
        end_pub = optimized_route[i + 1]
        
        start_coords = G.nodes[start_pub]['pos']
        end_coords = G.nodes[end_pub]['pos']
        
        route, _ = get_osrm_route(start_coords, end_coords)
        if route:
            route_latlong = [(coord[1], coord[0]) for coord in route]
            folium.PolyLine(locations=route_latlong, color='red', weight=2.5, opacity=0.8).add_to(m_optimized)

    map_file_path = 'static/optimized_route_map.html'
    m_optimized.save(map_file_path)


    return render_template('optimized_map.html', map_file_path=map_file_path, optimized_distance=optimized_distance)

if __name__ == '__main__':
    app.run(debug=True)
