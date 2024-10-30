import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import math
from queue import PriorityQueue
import copy
import numpy as np

PLOT_GRAPH = False
COMPILE_VIDEO = False

MAX_WEIGHT = 6

PAUSE_TIMER = 0.001

NODE_VISITED_COLOR = 'yellow'
NODE_SHORTESTPATH_COLOR = 'red'
EDGE_VISITED_COLOR = 'blue'
EDGE_SHORTESTPATH_COLOR = 'red'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~CLASS DECLARATION~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Edge:
    def __init__(self, start, end, weight):
        self.start = start
        self.end = end
        self.weight = weight

class Node:
    def __init__(self, id, x, y, edges, parent, heuristic):
        self.id = id
        self.x = x
        self.y = y
        self.edges = edges
        self.distance = float("inf")
        self.parent = parent
        self.heuristic = heuristic

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~READING INPUTS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Read the total number of nodes, start, end, and edge information from the inputfil
def read_inputfile():

    #open input file for reading and place all lines in a list
    with open('input.txt', 'r') as file:
        txt = file.readlines()

    # 
    total_nodes = int(txt[0].strip())
    start = int(txt[1].strip())
    end = int(txt[2].strip())

    edges_list = []
    for line in txt[3:]:
        edge = [float(component) for component in line.split()]
        
        edges_list.append(Edge(int(edge[0]), int(edge[1]), edge[2]))

    return [edges_list, total_nodes, start, end]

# Read the coordinates or each nodes from the cooridnates file
def read_coordinatesfile(edges: Edge, start, end):
    coords = []

    with open('coords.txt', 'r') as file:
        txt = file.readlines()
        for line in txt:
            coords.append([float(component) for component in line.split()])
            
    
    node_lists = [[] for _ in range(6)]
    for i in range(coords.__len__()):
        edge_list = []
        for edge in edges: #this for loop (searching through all edges for every node) could be optimized
            if edge.start == i+1:
                edge_list.append(edge) 
        heuristic = math.hypot(coords[i][0] - coords[end-1][0], coords[i][1] - coords[end-1][1])
        node = Node(i+1, coords[i][0], coords[i][1], edge_list, None, heuristic) 
        for n in range(6):
            node_copy = copy.deepcopy(node)
            node_lists[n].append(node_copy)
    return node_lists




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~PLOTTING GRAPH~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Create Graph
def create_graph(edges, nodes):
    fig, ax = plt.subplots(2,3)

    for row in ax:
        for col in row:
            nodes = plot_nodes(col, nodes)
            plot_edges(col, edges, nodes)

    if(PLOT_GRAPH):
        plt.show(block=False)

    return ax, fig

#Plotting Nodes
def plot_nodes(ax, nodes):
    for n in nodes:
        ax.plot(n.x, n.y, 'o', color='black', zorder=2)
        
    
    # Mark the End node and start node with respective colors

    ax.plot(nodes[end-1].x, nodes[end-1].y, 'o', color=NODE_SHORTESTPATH_COLOR, zorder=4)
    ax.plot(nodes[start-1].x, nodes[start-1].y, 'o', color=NODE_VISITED_COLOR, zorder=4)
    
    return nodes

#Plotting Edges
def plot_edges(ax, edges, coordinates):

    for edge in edges:
        x_coords = [coordinates[ edge.start -1 ].x, coordinates[ edge.end -1 ].x]
        y_coords = [coordinates[ edge.start -1].y, coordinates[ edge.end -1].y]
        (ax.plot(x_coords, y_coords, color="black", linewidth=2, zorder=1))





#~~~~~~~~~~~~~~~~~~~~Weighted A*~~~~~~~~~~~~~~

def a_step_handler(node_lists, start, end, ax):

    open_lists = []
    for i in range(6): 
        node_lists[i][start-1].distance = 0
        open_lists.append(node_lists[i][:])  # Append a copy of each list

    iterations = [0,0,0,0,0,0]

    #while open_dikstras and open_a_star_1 and open_a_star_2 and open_a_star_3 and open_a_star_4  and open_a_star_5:
    while not all(not row for row in open_lists):
        done = [False,False,False,False,False,False]
        for i in range(6):
            done[i] = a_star_one_step(node_lists[i], open_lists[i],start,end,ax[int(i/3),int(i%3)],i)
            if(done[i] == False):
                iterations[i] += 1

        if(PLOT_GRAPH):
            plt.pause(PAUSE_TIMER)
            plt.draw()
        
        if all(done):
            break
    
    # Backtrack

    backtrack_nodes = []
    shortestpath_id_lists = [[] for _ in range(6)]
    shortestpath_weight_lists = [[] for _ in range(6)]

    for i in range(6):
         backtrack_nodes.append(node_lists[i][end-1])
         shortestpath_id_lists[i].insert(0, backtrack_nodes[i].id)
         shortestpath_weight_lists[i].insert(0, round(backtrack_nodes[i].distance,2))


    while not all(node.parent is None for node in backtrack_nodes):
        for i in range(6):
            node_lists[i], backtrack_nodes[i], shortestpath_id_lists[i], shortestpath_weight_lists[i] = backtrack_onestep(node_lists[i], 
                backtrack_nodes[i], shortestpath_id_lists[i], shortestpath_weight_lists[i], ax[int(i/3),int(i%3)])
        
        if(PLOT_GRAPH):
            plt.pause(PAUSE_TIMER)
            plt.draw()

    return shortestpath_id_lists, shortestpath_weight_lists, iterations
    
def a_step_handler_video(node_lists, start, end, ax, fig):
    metadata = dict(title='012155624', artist= 'Peter Conant')
    writer = FFMpegWriter(fps=15, metadata=metadata)


    with writer.saving(fig, "012155624.mp4", 100):

        # Save the First Frame
        writer.grab_frame()

        open_lists = []
        for i in range(6): 
            node_lists[i][start-1].distance = 0
            open_lists.append(node_lists[i][:])  # Append a copy of each list

        iterations = [0,0,0,0,0,0]

        #while open_dikstras and open_a_star_1 and open_a_star_2 and open_a_star_3 and open_a_star_4  and open_a_star_5:
        while not all(not row for row in open_lists):
            done = [False,False,False,False,False,False]
            for i in range(6):
                done[i] = a_star_one_step(node_lists[i], open_lists[i],start,end,ax[int(i/3),int(i%3)],i)
                if(done[i] == False):
                    iterations[i] += 1

            if(PLOT_GRAPH):
                plt.pause(PAUSE_TIMER)
                plt.draw()
            
            writer.grab_frame()
            if all(done):
                break
        
        # Backtrack

        backtrack_nodes = []
        shortestpath_id_lists = [[] for _ in range(6)]
        shortestpath_weight_lists = [[] for _ in range(6)]

        for i in range(6):
            backtrack_nodes.append(node_lists[i][end-1])
            shortestpath_id_lists[i].insert(0, backtrack_nodes[i].id)
            shortestpath_weight_lists[i].insert(0, round(backtrack_nodes[i].distance, 2))


        while not all(node.parent is None for node in backtrack_nodes):
            for i in range(6):
                node_lists[i], backtrack_nodes[i], shortestpath_id_lists[i], shortestpath_weight_lists[i] = backtrack_onestep(node_lists[i], 
                    backtrack_nodes[i], shortestpath_id_lists[i], shortestpath_weight_lists[i], ax[int(i/3),int(i%3)])
            
            if(PLOT_GRAPH):
                plt.pause(PAUSE_TIMER)
                plt.draw()
            writer.grab_frame()

        return shortestpath_id_lists, shortestpath_weight_lists, iterations

# Weighted A* Algoirthm with (depending on golbal boolean) graph plotting
def a_star_one_step(nodes, open, start, end, ax, weight):

    # assign the node with the lowest distance to current
    current = min(open, key=lambda node: node.distance + (weight * node.heuristic)) 

    # minimum node from open is the end node, then we have found the shortest path for end n
    if(current.id == end):
        ax.plot(current.x, current.y, 'o', color='green', zorder= 4)
        return True

    ax.plot(current.x, current.y, 'o', color=NODE_VISITED_COLOR, zorder= 4)

    #
    for edge in current.edges:
        neighbour = nodes[edge.end-1]
        new_distance = current.distance + edge.weight

        x_coords = nodes[edge.start-1].x, nodes[edge.end-1].x
        y_coords = nodes[edge.start-1].y, nodes[edge.end-1].y

        ax.plot(x_coords, y_coords, color=EDGE_VISITED_COLOR, linewidth=2, zorder= 3)

        if new_distance < neighbour.distance:
            neighbour.distance = new_distance
            neighbour.parent = current

    open.remove(current) 
    return False


def backtrack_onestep(nodes, backtrack_node, shortestpath_ids, shortestpath_weights, ax):

    if(backtrack_node.parent is None):
        return nodes, backtrack_node, shortestpath_ids, shortestpath_weights
    
    backtrack_node = backtrack_node.parent

    shortestpath_ids.insert(0, backtrack_node.id)
    shortestpath_weights.insert(0, round(backtrack_node.distance, 2))

    x_coords = nodes[shortestpath_ids[0]-1].x, nodes[shortestpath_ids[1]-1].x
    y_coords = nodes[shortestpath_ids[0]-1].y, nodes[shortestpath_ids[1]-1].y
    ax.plot(x_coords, y_coords, color=NODE_SHORTESTPATH_COLOR, linewidth=2, zorder= 3)

    ax.plot(backtrack_node.x, backtrack_node.y, 'o', color=NODE_SHORTESTPATH_COLOR, zorder= 4)

    return [nodes, backtrack_node, shortestpath_ids, shortestpath_weights]




#~~~~~~~~~MAIN~~~~~~~~~~~~~

#readinputs
[edges, total_nodes, start, end] = read_inputfile()
node_lists = read_coordinatesfile(edges, start, end)

# If you wish to test other start/end points on the graph, enter those values here
end = end
start = start

weight = 1

#Plot Graph
ax, fig= create_graph(edges, node_lists[0])

#Run Dijkstrad
if(COMPILE_VIDEO):
    [shortestpath_id_lists, shortestpath_weight_lists, iterations] = a_step_handler_video(node_lists, start, end, ax, fig)
else:
    [shortestpath_id_lists, shortestpath_weight_lists, iterations] = a_step_handler(node_lists, start, end, ax)


#Print Output File
with open('012155624.txt', 'w') as file:
    for i in range(6):
        print("Shortest path for weight " + str(i) + ":" + str(shortestpath_id_lists[i]))
        print("Shortest path distance values for weight " + str(i) + ":" + str(shortestpath_weight_lists[i]))
        for id in shortestpath_id_lists[i]:
            file.write(str(id) + ' ')
        file.write('\n')
        for weight in shortestpath_weight_lists[i]:
            file.write(str(weight) + ' ')
        file.write('\n')

print('Number of iterations for each algorithm:' + str(iterations))