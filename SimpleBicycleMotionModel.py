import math


'''-------------------------SimpleBicycleMotionModel Help-------------------------------------

 https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html

 Prediction model 1

 Prediction model 2

 Prediction model 3
 steer angle (positive, negative sign)
 x, y global axes
'''


#-------------------------------------------------------------------------------------------

def predict_motion_model_1(v, steer_angle, time_diff, x, y, angle):
  L = 0.36

  x_dot = v * math.cos(angle)
  y_dot = v * math.sin(angle)
  theta_dot = (v * math.tan(-steer_angle)) / L

  new_x = x + (x_dot * time_diff)
  new_y = y - (y_dot * time_diff)
  new_angle = angle + theta_dot * time_diff

  #print("After calculate: ", "x: ", new_x, "y: ", new_y, "angle: ", new_angle)
  return new_x, new_y, new_angle

#-------------------------------------------------------------------------------------------

def predict_motion_model_3(v, steer_angle, time_diff, x, y, angle):
  Lr = 0.18
  L = 0.36

  b = math.atan((Lr*math.tan(-steer_angle))/L)
  x_dot = v * math.cos(angle + b)
  y_dot = v * math.sin(angle + b)
  theta_dot = (v * math.tan(-steer_angle) * math.cos(b)) / L

  new_x = x + (x_dot * time_diff)
  new_y = y - (y_dot * time_diff)
  new_angle = angle + theta_dot * time_diff

  #print("After calculate: ", new_x, new_y, new_angle)
  return new_x, new_y, new_angle

#-------------------------------------------------------------------------------------------

def predict_motion_model_2(v, steer_angle, time_diff, x, y, angle):
  L = 0.36

  x_dot = v * math.cos(angle - steer_angle)
  y_dot = v * math.sin(angle - steer_angle)
  theta_dot = (v * math.sin(-steer_angle)) / L

  new_x = x + (x_dot * time_diff)
  new_y = y - (y_dot * time_diff)
  new_angle = angle + theta_dot * time_diff

  #print("After calculate: ", "x: ", new_x, "y: ", new_y, "angle: ", new_angle)
  return new_x, new_y, new_angle

#-------------------------------------------------------------------------------------------