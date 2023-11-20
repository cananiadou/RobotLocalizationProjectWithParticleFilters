import os
import json
import numpy as np
import math
import ReadLogFile
from ReadLogFile import read_file
import cv2
import matplotlib.pyplot as plt
import scipy.stats as stats
from rospy_message_converter import message_converter
import SimpleBicycleMotionModel
from SimpleBicycleMotionModel import *
import networkx as nx
import random
import statistics
from ReportLab import *
import glob
import argparse


''' ---------------------------RobotLocalizationWithParticleFilters Help-------------------------------



'''
#------------------------------------------------------------------------------------
# script arguments
# Define a custom validation function
def check_index(value):
    ivalue = int(value)
    if ivalue < 0 or ivalue>=10:
        raise argparse.ArgumentTypeError(f"{value} Index is incorrect")
    return ivalue

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Robot Localization with Particle Filters')

# Add named arguments
# parser.add_argument('-folder', help='Specify the path folder.')
parser.add_argument('-particles', required=True, help='Number of particles')
parser.add_argument('-index', required=True, type=check_index, help='0: ramp2roundabout, 1: roundabout2ramp, 2: parking2roundabout, '
                    '3: roundabout2parking, 4: highwayUp, 5: highwayDown, 6: bumpy, 7: manualIntersections, '
                    '8: ramp2rampX3, 9: highway2roundabout')
parser.add_argument('-discarded', action='store_true', help='Use discarded model')
parser.add_argument('-visualize_loc', action='store_true', help='Visualize localization nodes')
parser.add_argument('-delete_old', action='store_true', help='Delete old files')
args = parser.parse_args()

# Access the values of the arguments


# ----------------------------------Global variables---------------------------------
ALMOST_ZERO = 1e-5
PARTICLES_NUM = args.particles
GROWTH_SCALE = 0.005
LANE_LENGTH = 0.35
RESAMPLE_RATIO = 0.75
INDEX = 8
vel_std = 0.04
steer_std = 0.2

IMAGE_PATH = "C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Competition_map.png"
IMAGE = cv2.imread(IMAGE_PATH)
GRAPH_PATH = 'C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Competition_track.graphml'
G = nx.read_graphml(GRAPH_PATH)
max_x = 14.67
max_y = 15.0
h, w, _ = IMAGE.shape
factor_x = w / max_x
factor_y = h / max_y
# -------------------------------End of Global variables-----------------------------

#----------------------------------------------------------------------------------------------------
ramp2roundabout_nodes = ("62","148","149","150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167",
         "168", "169", "170", "171", "198", "199", "200", "201", "202", "203", "204", "205", "206","207","208", "209", "210", "211", "212", "213", "214",
         "215", "216", "217", "218", "219", "220", "221", "222", "223", "224", "225", "226", "227", "228", "229", "230")
#----------------------------------------------------------------------------------------------------
parking2roundabout_nodes = ("160", "161", "162", "163", "164", "165", "166", "167",
                           "168", "169", "170", "171", "198", "199", "200", "201", "202", "203", "204", "205", "206","207","208", "209", "210", "211", "212", "213", "214",
                           "215", "216", "217", "218", "219", "220", "221", "222", "223", "224", "225", "226", "227", "228", "229", "230")
#----------------------------------------------------------------------------------------------------
roundabout2ramp_nodes = ("231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", "242", "243", "244", "245", "246",
                         "247", "248", "249", "250", "251", "252", "253", "254", "255", "256", "257", "258", "259", "260", "261", "262",
                         "263", "264", "265", "266", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183",
                         "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194", "195", "196", "197", "63")
#--------------------------------------------------------------------------------------------------
roundabout2parking_nodes = ("231", "232", "233", "234", "235", "236", "237", "238", "239", "240", "241", "242", "243", "244", "245", "246",
                            "247", "248", "249", "250", "251", "252", "253", "254", "255", "256", "257", "258", "259", "260", "261", "262",
                            "263", "264", "265", "266", "172", "173", "174", "175", "176", "177", "178", "179", "180", "181", "182")
#----------------------------------------------------------------------------------------------------
bumpy_nodes = ("426", "427", "428", "429", "430", "431", "432", "433", "434", "435", "436", "437", "438", "439", "440", "441", "442", "443",
               "444", "445", "446", "447", "448", "449", "450", "451", "452", "453", "454", "455", "456", "457", "458", "459", "460", "461",
               "462", "463", "464", "465", "466", "467")
#----------------------------------------------------------------------------------------------------
highwayUp_nodes = ("49", "308", "309", "310", "311", "312", "313", "314", "315", "316", "317", "318", "319", "320", "321", "322", "323", "324",
                   "325", "326", "327", "328", "329", "330", "331", "332", "333", "334", "335", "336", "337", "338", "339", "340", "341",
                   "342")
#----------------------------------------------------------------------------------------------------
highwayDown_nodes = ("343", "344", "345", "346", "347", "348", "349", "350", "351", "352", "353", "354", "355", "356", "357", "358", "359",
                     "360", "361", "362", "363", "364", "365", "366", "367", "368", "369", "370", "371", "372", "373", "374")
#----------------------------------------------------------------------------------------------------
manualIntersections_nodes = ("78", "87", "45", "46", "40", "90", "54", "55", "51", "104", "105", "106", "36", "37", "31", "114", "115",
                    "116", "117", "118", "27", "28", "22", "288", "289", "290", "291", "292", "293", "294", "295", "296", "297",
                    "298", "299", "300", "301", "302", "303", "304", "343", "344", "345", "346", "347", "348", "349", "350",
                    "351", "352", "353", "354", "355", "356", "357", "358", "359", "360", "361", "362", "363", "364", "365",
                    "366", "367", "368", "369", "370", "371", "372", "373", "374", "51", "104", "105", "106", "36", "37", "33",
                    "113", "6", "9", "8", "139", "140", "141", "142", "143", "15", "19", "13", "145", "61", "65", "62", "148","149",
                    "150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165",
                    "166", "167", "168", "169", "170", "171", "198", "199", "200", "201", "202", "203", "204", "205", "206","207","208",
                    "209", "210"
      )
#----------------------------------------------------------------------------------------------------
ramp2ramp_nodes = ("149","150", "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161", "162", "163", "164", "165", "166", "167",
         "168", "169", "170", "171", "198", "199", "200", "201", "202", "203", "204", "205", "206","207","208", "209", "210", "211", "212", "213", "214",
         "215", "216", "217", "218", "219", "220", "221", "222", "223", "224", "225", "226", "227", "228", "229", "230", "267", "268", "269", "270", "271",
        "272", "273", "274", "275", "276", "277", "278", "279", "280", "281", "282", "283", "284", "285", "286", "287", "23", "24", "147", "18", "19", "13",
        "145", "61")
#----------------------------------------------------------------------------------------------------
highway2roundabout_nodes = ("304", "343", "344", "345", "346", "347", "348", "349", "350", "351", "352", "353", "354", "355", "356", "357", "358", "359",
                     "360", "361", "362", "363", "364", "365", "366", "367", "368", "369", "370", "371", "372", "373", "374", "51", "104", "105",
                            "106", "36", "31", "114", "115", "116", "117", "118", "27", "22", "288", "289", "290", "291", "292", "293", "294",
                            "295", "296", "297", "298", "299", "300", "301", "302", "303")
#----------------------------------------------------------------------------------------------------

files = [
"C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Data_23_09\\ros_ramp2roundaboutTest\\",
"C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Data_23_09\\ros_roundabout2rampTest\\",
"C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Data_23_09\\ros_parking2roundaboutTest\\",
"C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Data_23_09\\ros_roundabout2parkingTest\\",
"C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Data_23_09\\ros_highwayUpTest\\",
"C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Data_23_09\\ros_highwayDownTest\\",
"C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Data_23_09\\ros_bumpyTest\\",
"C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Data_18_11\\ros_manual_intersectionsTest\\",
"C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Data_18_11\\ros_full_ramp2ramp_x3Test\\",
"C:\\Users\\xrist\\Documents\PostGraduate\\Thesis\\Data_18_11\\ros_full_highway_to_roundabout_almost_cyrcleTest\\"
]

nodes = [
  ramp2roundabout_nodes,
  roundabout2ramp_nodes,
  parking2roundabout_nodes,
  roundabout2parking_nodes,
  highwayUp_nodes,
  highwayDown_nodes,
  bumpy_nodes,
  manualIntersections_nodes,
  ramp2ramp_nodes,
  highway2roundabout_nodes
]
#----------------------------------------------------------------------------------------------------
class Report:
  def __init__(self, num_particles: int, growth_scale : float, resample_ratio : float, path, discarded: bool, vel_std: float, steer_std: float):
    self.num_particles = num_particles
    self.growth_scale = growth_scale
    self.resample_ratio = resample_ratio
    self.path = path
    self.resample_num = 0
    self.weights_mean_bn = []
    self.weights_mean_an = []
    self.best_weights = []
    self.robot_posxs = []
    self.robot_posys = []
    self.est_robot_posxs = []
    self.est_robot_posys = []
    self.discarded = discarded
    self.vel_std = vel_std
    self.steer_std = steer_std

  def add_nums(self, resample_num, sample_num):
    self.resample_num = resample_num
    self.sample_num = sample_num

#----------------------------------------------------------------------------------------------------
class Node:
  def __init__(self, x: float, y: float):
    self.x = x
    self.y = y

#----------------------------------------------------------------------------------------------------
class Position:
  def __init__(self, x: float, y: float, angle: float):
    self.x = x
    self.y = y
    self.angle = angle

  @classmethod
  def fromNode(cls, node: Node, angle: float):
    return cls(node.x, node.y, angle)

#----------------------------------------------------------------------------------------------------

class Particle:
  def __init__(self, x: float, y: float, angle: float, weight: float):
    self.x: float = x
    self.y: float = y
    self.angle: float = angle
    self.weight: float = weight
    self.factor_x, self.factor_y = factor_x, factor_y
    self.color=(255, 0, 0)
    self.min_dist = 15.0
    self.lane_length = LANE_LENGTH
    self.dist = 0

  def print(self):
    print(self.x, self.y, self.angle, self.weight)

  def visualize(self, is_best=False):
    px = int(self.x * self.factor_x)
    py = int(self.y * self.factor_y)

    if is_best:
      cv2.circle(IMAGE, (px, py), color=(0, 255, 0), radius=5, thickness=1)
    else:
      cv2.circle(IMAGE, (px, py), color=(0, 255, 0), radius=5, thickness=1)
    #cv2.imwrite(pic, IMAGE)

  def update_values(self, x, y, angle):
    self.x = x
    self.y = y
    self.angle = angle

  def update_weight(self):
    node1, node2 = self.find_closest_two_nodes()
    n1 = Node(G.nodes[node1]["x"], G.nodes[node1]["y"])
    n2 = Node(G.nodes[node2]["x"], G.nodes[node2]["y"])
    A, B, C = self.calculate_coefficients_of_line(n1, n2)
    self.dist = self.calc_perpendicular_dist_to_line(A, B, C)
    if self.dist > self.lane_length/2:
      self.weight = 1e-10
    else:
      self.weight = 1 - self.dist/(self.lane_length/2)

  def find_closest_two_nodes(self):
    closest_node = nodes[args.index][0]
    for node in nodes[args.index]:
      x1 = G.nodes[node]["x"]
      y1 = G.nodes[node]["y"]
      dist = self.calc_dist(Node(x1, y1))
      if dist < self.min_dist:
        closest_node = node
        self.min_dist = dist
    next_node = list(G.adj[closest_node])[0]
    self.min_dist = 15.0
    return closest_node, next_node

  def calc_dist(self, node):
    return math.sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2)

  def calculate_coefficients_of_line(self, n1, n2):
    A = n2.y - n1.y
    B = n1.x - n2.x
    C = (n2.x * n1.y) - (n1.x * n2.y)
    return A, B, C

  def calc_perpendicular_dist_to_line(self, A, B, C):
    distance = abs(A * self.x + B * self.y + C) / math.sqrt(A ** 2 + B ** 2)
    return distance

#----------------------------------------------------------------------------------------------------

def initialize_particles_gaussian(init_x, init_y, init_angle, growth_scale, num_particles = PARTICLES_NUM):
  x_list = list(np.random.normal(loc=init_x, scale=growth_scale, size=num_particles - 1))
  y_list = list(np.random.normal(loc=init_y, scale=growth_scale, size=num_particles - 1))
  angle_list = list(np.random.normal(loc=init_angle, scale=0.01, size=num_particles - 1))
  initial_weight = 1.0 / float(num_particles)

  particles = list()
  for x, y, angle in zip(x_list, y_list, angle_list):
    particles.append(Particle(x, y, angle, initial_weight))
  particles.append(Particle(init_x, init_y, init_angle, initial_weight))

  return particles

#----------------------------------------------------------------------------------------------------

def print_particles(header, particles):
  for particle in particles:
    particle.print()

#----------------------------------------------------------------------------------------------------

def get_loc_time(loc_dict_c, loc_dict, c):
  x, y, a = loc_dict_c[c]

  for key, value in loc_dict.items():
    if value[0] == x and value[1] == y and value[2] == a:
        return key

  raise ValueError("Value not found in the dictionary")
  return get_loc_time(loc_dict_c, loc_dict, len(loc_dict)-1)

#----------------------------------------------------------------------------------------------------

def get_measurements(line):
  line_data = line.split(",")
  time = float(line_data[0])
  velocity = float(line_data[1])
  steer_angle = float(line_data[2])

  return time, velocity, steer_angle

#----------------------------------------------------------------------------------------------------

def collect_localization_data(file):
  loc_dict = {}
  loc_dict_c = {}
  c: int = 0
  with open(file) as f:
    for line in f:
      line_data = line.split(",")
      time = float(line_data[0])
      x = float(line_data[1])
      y = float(line_data[2])
      angle = float(line_data[3])
      loc_dict[time] = (x, y, angle)
      loc_dict_c[c] = (x, y, angle)
      c += 1

  return loc_dict, loc_dict_c

# ----------------------------------------------------------------------------------------------------

def find_best_particle(particles):
  weights = [particle.weight for particle in particles]
  max_number = max(weights)

  # Find all indices of the maximum number in the list
  max_indices = [index for index, value in enumerate(weights) if value == max_number]

  if len(max_indices) > 1:
    print("More than one best particle")
    print("Length of max indices", len(max_indices), "max weight is", max_number)

  return particles[max_indices[0]]

# ----------------------------------------------------------------------------------------------------

def create_histograms(data1, data2):
  # Create a figure and axis
  fig, ax = plt.subplots()

  # Plot the histograms
  ax.hist(data1, bins=10, alpha=0.5, label='Histogram 1')
  ax.hist(data2, bins=10, alpha=0.5, label='Histogram 2')

  # Add labels and a legend
  ax.set_xlabel('particle index')
  ax.set_ylabel('Weight')
  ax.legend()

  # Show the plot
  plt.show()

# ----------------------------------------------------------------------------------------------------

def simple_resampling_algorithm(particles):
  N = len(particles)
  weights = [particle.weight for particle in particles]
  cumulative_sum = np.cumsum(weights)
  cumulative_sum[-1] = 1. # avoid round-off error
  indexes = np.searchsorted(cumulative_sum, np.random.random(N))

  #resample according to indexes
  new_particles = []
  for i in indexes:
    new_particles.append(Particle(particles[i].x, particles[i].y, particles[i].angle, particles[i].weight))

  assert len(particles) == len(new_particles), "Length of new particles is not the same"

  for particle in new_particles:
    particle.weight = 1.0 /float(N)

  return new_particles

# ----------------------------------------------------------------------------------------------------

def resample(particles):
  '''
  weights = [particle.weight for particle in particles]
  cum_sums = np.cumsum(weights).tolist()
  n = 0
  new_samples = []

  initial_weight = 1.0 / float(PARTICLES_NUM)
  while n < PARTICLES_NUM:
    u = np.random.uniform(1e-6, 1, 1)[0]
    m = 0
    while cum_sums[m] < u:
      m += 1
    new_samples.append(Particle(particles[m].x, particles[m].y, particles[m].angle, initial_weight))
    n += 1
  '''
  return simple_resampling_algorithm(particles)

#----------------------------------------------------------------------------------------------------

def should_resample(particles, resampling_threshold):
  sum_weights_squared = 0.0
  for particle in particles:
    sum_weights_squared += particle.weight ** 2

  return (1.0 / sum_weights_squared) < resampling_threshold

#----------------------------------------------------------------------------------------------------

def estimate(particles):
  x = 0.0
  y = 0.0
  angle = 0.0
  for particle in particles:
    x += particle.x * particle.weight
    y += particle.y * particle.weight
    angle += particle.angle * particle.weight
  node = Node(x=x, y=y)
  pos = Position.fromNode(node=node, angle=angle)
  return x, y

#----------------------------------------------------------------------------------------------------

def normalize(particles):
  sum_of_weights = 0.0
  for particle in particles:
    sum_of_weights += particle.weight

  #sum_of_weights += 1.e-300  # avoid round-off to zero
  if sum_of_weights < ALMOST_ZERO:
    print("Yes almost zero...")
    #question maybe here I should do something else?
    for particle in particles:
      particle.weight = 1.0 / float(len(particles))
  else:
    for particle in particles:
      particle.weight = particle.weight/sum_of_weights

# ----------------------------------------------------------------------------------------------------

def visualize(particles):
  for particle in particles:
      particle.visualize()

  cv2.namedWindow("Robot Localization with Particle Filters", cv2.WINDOW_NORMAL)
  cv2.imshow("Robot Localization with Particle Filters", IMAGE)

  if args.visualize_loc:
    while True:
    # Wait for a key event
      key = cv2.waitKey(1) & 0xFF

    # Check if the 'q' key is pressed to exit the loop and close the window
      if key == ord('q'):
        break
  else:
    cv2.waitKey(1)

  # cv2.destroyAllWindows()

# ----------------------------------------------------------------------------------------------------

def update_weights(particles):
  for particle in particles:
    particle.update_weight()

#-----------------------------------------------------------------------------------------------------

def create_gaussian_velocity(velocity, std):
  new_velocity = list(np.random.normal(loc=velocity, scale=std, size=PARTICLES_NUM - 1))
  new_velocity.append(velocity)
  return new_velocity

#-----------------------------------------------------------------------------------------------------

def create_gaussian_angle(angle, std):
  new_angle = list(np.random.normal(loc=angle, scale=std, size=PARTICLES_NUM - 1))
  new_angle.append(angle)
  return new_angle


# -----------------------------------------------------------------------------------------------------

def predict(particles, velocity, steer_angle, time_diff):
  #new_velocity = velocity + std[0]*np.random.randn(len(particles))
  #new_steer_angle = steer_angle + std[1]*np.random.randn(len(particles))
  new_velocity = create_gaussian_velocity(velocity, vel_std)
  new_steer_angle = create_gaussian_angle(steer_angle, steer_std)

  for i, particle in enumerate(particles):

    new_x, new_y, new_angle = SimpleBicycleMotionModel.predict_motion_model_2(new_velocity[i],
                                                                  new_steer_angle[i],
                                                                  time_diff,
                                                                  particle.x,
                                                                  particle.y,
                                                                  particle.angle)
    particle.update_values(new_x, new_y, new_angle)

#----------------------------------------------------------------------------------------------------

def apply_algorithm(particles, command_file, loc_dict_c, loc_dict, report):
  previous_time = 0
  counter = 0
  estimate_counter = 0
  num_resample = 0
  should_estimate = True
  tot_time = 0
  loc_time = 0

  with open(command_file) as f:
    for line in f:
      time, velocity, steer_angle_de = get_measurements(line)
      steer_angle = math.radians(steer_angle_de)
      time_diff = time - previous_time
      previous_time = time

      predict(particles, velocity, steer_angle, time_diff)

      update_weights(particles)
      best_particle = find_best_particle(particles)
      visualize(particles)

      weights = [particle.weight for particle in particles]
      report.weights_mean_bn.append(statistics.mean(weights))
      report.best_weights.append(best_particle.weight)

      normalize(particles)

      weights = [particle.weight for particle in particles]
      report.weights_mean_an.append(statistics.mean(weights))

      if tot_time > loc_time:
        should_estimate = True

      if should_estimate:
        x_est, y_est = estimate(particles)
        robot_posx, robot_posy, _ = loc_dict_c[estimate_counter]
        report.robot_posxs.append(robot_posx)
        report.robot_posys.append(-robot_posy)
        report.est_robot_posxs.append(x_est)
        report.est_robot_posys.append(-y_est)
        estimate_counter += 1
        should_estimate = False
        loc_time = get_loc_time(loc_dict_c, loc_dict, estimate_counter)

      if should_resample(particles, RESAMPLE_RATIO*PARTICLES_NUM):
        print("Resampling....")
        num_resample += 1
        particles = resample(particles)
        assert len(particles) == PARTICLES_NUM, "Length is not the Num of particles"
      counter += 1
      tot_time += time_diff

    report.add_nums(num_resample, counter)

#----------------------------------------------------------------------------------------------------

def delete_previous(main_folder):
  if args.delete_old:
    txt_extension = '*.txt'
    txt_files = glob.glob(f'{main_folder}/{txt_extension}')

    for txt_file in txt_files:
      try:
        os.remove(txt_file)
        print(f"Deleted: {txt_file}")
      except Exception as e:
        print(f"Error deleting {txt_file}: {e}")

#----------------------------------------------------------------------------------------------------

def read_new(main_folder):
  extension = '*.log'
  files = glob.glob(f'{main_folder}/{extension}')
  for filepath in files:
    command_file, loc_file = ReadLogFile.read_file(filepath)

  return command_file, loc_file

#----------------------------------------------------------------------------------------------------

if __name__ == '__main__':
  random.seed(1) # question comment this maybe?
  main_folder = files[args.index]
  delete_previous(main_folder)
  command_file, loc_file = read_new(main_folder)
  loc_dict, loc_dict_c = collect_localization_data(loc_file)

  if args.visualize_loc:
    particles = [Particle(val[0], val[1], val[2], 1) for val in loc_dict.values()]
    visualize(particles)
  else:
    init_x, init_y, init_angle = loc_dict[0.0]
    particles = initialize_particles_gaussian(init_x, init_y, init_angle, GROWTH_SCALE)
    visualize(particles)

    if args.discarded:
      command_file = main_folder + "discarded_command_measurements.txt"

    report = Report(PARTICLES_NUM, GROWTH_SCALE, RESAMPLE_RATIO, command_file, args.discarded, vel_std, steer_std)
    apply_algorithm(particles, command_file, loc_dict_c, loc_dict, report)
    create_report(report)
