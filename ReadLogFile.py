import os
import json
import math


''' ---------------------------------ReadLogFile Help---------------------------------------

 In this script a log file with measurements that gathered from gazebo, is processed

 These measurements are command, localization and lane points topics of gazebo
 After some processing (time correction and data syncing),
 two files are produced
 one for the command data
 and another for the localization data

'''

#-----------------------------------------------------------------------------------------

def create_command_log_file(data):
  log_data = ""
  time = data['timestamp']
  action = data['action']
  log_data += str(action) + "," + str(time) + ","
  if 'speed' in data:
    speed = data['speed']
    log_data += " "
    log_data += str(speed)
  elif 'steerAngle' in data:
    steer_angle = data['steerAngle']
    log_data += " "
    log_data += str(steer_angle)
  log_data += "\n"
  return log_data

#-----------------------------------------------------------------------------------------

def correct_command_time_data(file):
  fix_data = ""
  with open(file) as f:
    first_time = False
    for line in f:
      line_data = line.split(",")

      if not first_time:
       first_time = True
       init_time = line_data[1]

      fix_time = float(line_data[1]) - float(init_time)
      fix_data += line_data[0] + ", " + str(fix_time) + ", " + line_data[2]

  return fix_data

#-----------------------------------------------------------------------------------------

def sync_command_data(file):
  fix_data = ""
  first_speed = False
  first_steer = False
  speed = ""
  steer_angle = ""

  with open(file) as f:
    for line in f:
      line_data = line.strip().split(",")
      action = line_data[0]

      if action == "1":  # speed
        speed = line_data[2]
        if not first_speed:
          first_speed = True

      if action == "2":  # steer_angle
        steer_angle = line_data[2]
        if not first_steer:
          first_steer = True

      if not first_speed or not first_steer:
        continue
      else:
        fix_data += line_data[1] + ", " + speed + ", " + steer_angle + "\n"

  return fix_data

#-----------------------------------------------------------------------------------------

def correct_command_file(file, directory):
  fixed_time_data = correct_command_time_data(file)
  fix_file = os.path.join(directory, "command_fixed.txt")
  with open(fix_file, 'w') as f:
    f.write(fixed_time_data)

  synced_data = sync_command_data(fix_file)
  sync_file = os.path.join(directory, "command_measurements.txt")
  with open(sync_file, 'w') as f:
   f.write(synced_data)

  return sync_file

#-----------------------------------------------------------------------------------------

def correct_loc_file(file, directory):
  fix_data = ""
  with open(file) as f:
    first_time = False
    for line in f:
      line_data = line.strip().split(",")

      if not first_time:
       first_time = True
       init_time = line_data[0]

      fix_time = float(line_data[0]) - float(init_time)
      fix_data += str(fix_time) + ", " + line_data[1] + ", " + line_data[2] + ", " + line_data[3] + ", " + str(math.degrees(float(line_data[3]))) + "\n"

  fixed_file = os.path.join(directory, "localization_measurements.txt")
  with open(fixed_file, 'w') as f:
    f.write(fix_data)
  return fixed_file

#-----------------------------------------------------------------------------------------

def create_loc_log_file(data):
  log_data = ""
  time = data['timestamp']
  posA = data['posA']
  posB = data['posB']
  rotA = data['rotA']
  log_data += str(time) + ", "
  log_data += str(posA)
  log_data += ", "
  log_data += str(posB)
  log_data += ", "
  log_data += str(rotA)
  log_data += "\n"
  return log_data

#-----------------------------------------------------------------------------------------

def read_file(filepath):
  com_data = ""
  loc_data = ""
  directory = os.path.dirname(filepath)

  with open(filepath, 'r') as log_file:
    for line in log_file:
      data = line.strip('\n').split('INFO')[1].lstrip(' -')
      data = data.replace("'", "\"")
      dict_data = json.loads(data)
      header = dict_data['header']
      if header == 'COM':
        com_data += create_command_log_file(dict_data)
      elif header == 'LOC':
        loc_data += create_loc_log_file(dict_data)

  loc_file = os.path.join(directory, "data_loc.txt")
  with open(loc_file, 'w') as f:
    f.write(loc_data)
  new_loc_file = correct_loc_file(loc_file, directory)

  raw_file = os.path.join(directory, "command_raw.txt")
  with open(raw_file, 'w') as f:
    f.write(com_data)
  new_command_file = correct_command_file(raw_file, directory)

  return new_command_file, new_loc_file

#-----------------------------------------------------------------------------------------