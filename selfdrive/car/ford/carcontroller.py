from os import path
from openpilot.common.params import Params
import json

import numpy as np
from cereal import car, log
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.numpy_fast import clip, interp
from openpilot.common.realtime import DT_CTRL
from opendbc.can.packer import CANPacker
from openpilot.selfdrive.car import apply_std_steer_angle_limits
from openpilot.selfdrive.car.ford import fordcan
from openpilot.selfdrive.car.ford.values import CANFD_CAR, CarControllerParams
from openpilot.selfdrive.controls.lib.drive_helpers import V_CRUISE_MAX, CONTROL_N
from openpilot.selfdrive.modeld.constants import ModelConstants

LongCtrlState = car.CarControl.Actuators.LongControlState
VisualAlert = car.CarControl.HUDControl.VisualAlert
LaneChangeState = log.LaneChangeState

def apply_ford_curvature_limits(apply_curvature, apply_curvature_last, current_curvature, v_ego_raw):
  # No blending at low speed due to lack of torque wind-up and inaccurate current curvature
  if v_ego_raw > 9:
    apply_curvature = clip(apply_curvature, current_curvature - CarControllerParams.CURVATURE_ERROR,
                           current_curvature + CarControllerParams.CURVATURE_ERROR)

  # Curvature rate limit after driver torque limit
  apply_curvature = apply_std_steer_angle_limits(apply_curvature, apply_curvature_last, v_ego_raw, CarControllerParams)

  return clip(apply_curvature, -CarControllerParams.CURVATURE_MAX, CarControllerParams.CURVATURE_MAX)

def hysteresis(current_value, old_value, target, stdDevLow: float, stdDevHigh: float):
  if target - stdDevLow < current_value < target + stdDevHigh:
    result = old_value
  elif current_value <= target - stdDevLow:
    result = 1
  elif current_value >= target + stdDevHigh:
    result = 0

  return result

def actuators_calc(self, brake):
  ts = self.frame * DT_CTRL
  brake_actuate = hysteresis(brake, self.brake_actuate_last, self.brake_actutator_target, self.brake_actutator_stdDevLow, self.brake_actutator_stdDevHigh)
  self.brake_actuate_last = brake_actuate

  precharge_actuate = hysteresis(brake, self.precharge_actuate_last, self.precharge_actutator_target, self.precharge_actutator_stdDevLow, self.precharge_actutator_stdDevHigh)
  if precharge_actuate and not self.precharge_actuate_last:
    self.precharge_actuate_ts = ts
  elif not precharge_actuate:
    self.precharge_actuate_ts = 0

  if (
      precharge_actuate and 
      not brake_actuate and
      self.precharge_actuate_ts > 0 and 
      brake > (self.precharge_actutator_target - self.precharge_actutator_stdDevLow) and 
      (ts - self.precharge_actuate_ts) > (200 * DT_CTRL) 
    ):
    precharge_actuate = False

  self.precharge_actuate_last = precharge_actuate

  return precharge_actuate, brake_actuate

def set_ramp_type(lat_accel):
    """
    Determine the ramp type based on lateral acceleration (lat_accel).

    Parameters:
    - lat_accel (float): The lateral acceleration.

    Returns:
    - int: The ramp type (0 for Slow, 1 for Medium, 2 for Fast).
    """
    # Slow ramp type for lat_accel within [-0.06, 0.06]
    if -0.6 <= lat_accel <= 0.6:
        return 0  # Slow
    
    # Medium ramp type for lat_accel outside the slow range but within [-1.2, 1.2]
    elif (-1.2 < lat_accel < -0.6) or (0.6 < lat_accel < 1.2):
        return 1  # Medium
    
    # Fast ramp type for everything else
    else:
        return 2  # Fast


class CarController:
  def __init__(self, dbc_name, CP, VM):
    self.CP = CP
    self.VM = VM
    self.packer = CANPacker(dbc_name)
    self.CAN = fordcan.CanBus(CP)
    self.frame = 0

    self.apply_curvature_last = 0
    self.main_on_last = False
    self.lkas_enabled_last = False
    self.steer_alert_last = False
    self.brake_actuate_last = 0
    self.precharge_actuate_last = 0
    self.precharge_actuate_ts = 0
    self.json_data = {}
    jsonFile = path.join(path.dirname(path.abspath(__file__)), "tuning.json")
    if path.exists(jsonFile):
      vehFingerprint = "FORD F-150 14TH GEN" # Get the fingerprint to use in the JSON here
      f = open(jsonFile)
      self.json_data = json.load(f)
      if vehFingerprint in self.json_data and self.json_data[vehFingerprint] is not None:
        self.json_data = self.json_data[vehFingerprint]
      f.close()

    self.testing_active = True
    if 'testing_active' in self.json_data:
      self.testing_active = self.json_data['testing_active']

    self.brake_actutator_target = -0.1
    if 'brake_actutator_target' in self.json_data:
      self.brake_actutator_target = self.json_data['brake_actutator_target']

    # Activates at self.brake_actutator_target - self.brake_actutator_stdDevLow
    # Default: -0.5
    self.brake_actutator_stdDevLow = 0.2
    if 'brake_actutator_stdDevLow' in self.json_data:
      self.brake_actutator_stdDevLow = self.json_data['brake_actutator_stdDevLow']

    # Deactivates at self.brake_actutator_target + self.brake_actutator_stdDevHigh
    # Default: 0
    self.brake_actutator_stdDevHigh = 0.1
    if 'brake_actutator_stdDevHigh' in self.json_data:
      self.brake_actutator_stdDevHigh = self.json_data['brake_actutator_stdDevHigh']

    self.precharge_actutator_target = -0.1
    if 'precharge_actutator_target' in self.json_data:
      self.precharge_actutator_target = self.json_data['precharge_actutator_target']

    # Activates at self.precharge_actutator_target - self.precharge_actutator_stdDevLow
    # Default: -0.25
    self.precharge_actutator_stdDevLow = 0.1
    if 'precharge_actutator_stdDevLow' in self.json_data:
      self.precharge_actutator_stdDevLow = self.json_data['precharge_actutator_stdDevLow']

    # Deactivates at self.precharge_actutator_target + self.precharge_actutator_stdDevHigh
    # Default: 0
    self.precharge_actutator_stdDevHigh = 0.1
    if 'precharge_actutator_stdDevHigh' in self.json_data:
      self.precharge_actutator_stdDevHigh = self.json_data['precharge_actutator_stdDevHigh']

    self.brake_0_point = 0
    if 'brake_0_point' in self.json_data:
      self.brake_0_point = self.json_data['brake_0_point']

    self.brake_converge_at = -1.5
    if 'brake_converge_at' in self.json_data:
      self.brake_converge_at = self.json_data['brake_converge_at']

    self.brake_clip = self.brake_actutator_target - self.brake_actutator_stdDevLow

    # Deactivates at self.precharge_actutator_target + self.precharge_actutator_stdDevHigh
    # Default: 0
    self.target_speed_multiplier = 1
    if 'target_speed_multiplier' in self.json_data:
      self.target_speed_multiplier = self.json_data['target_speed_multiplier']
    
    # for computing other lat control inputs
    self.look_ahead_v = [0.8, 1.8] # how many seconds in the future to look ahead in [0, ~2.1] in 0.1 increments
    self.look_ahead_bp = [9.0, 35.0] # corresponding speeds in m/s in [0, ~40] in 1.0 increments
    # precompute time differences between ModelConstants.T_IDXS
    self.t_diffs = np.diff(ModelConstants.T_IDXS)
    self.desired_curvature_rate_scale = -0.07 # determined in plotjuggler to best match with `LatCtlCrv_NoRate2_Actl`
    self.future_lookup_time_diff = 0.5
    self.future_lookup_time = CP.steerActuatorDelay
    self.future_curvature_time_v = [self.future_lookup_time, self.future_lookup_time_diff + self.future_lookup_time] # how many seconds in the future to use predicted curvature
    self.future_curvature_time_bp = [5.0, 30.0] # corresponding speeds in m/s in [0, ~40] in 1.0 increments
    self.rate_future_time = 0.3 # how many seconds in the future for path offset rate date lookup
    self.path_offset_rate_position_lat_accel_adjust_scale = 0.007
    self.path_offset_rate_laneline_lat_accel_adjust_scale = 0.015
    self.path_offset_rate_laneline = FirstOrderFilter(0.0, 0.1, 0.05)
    self.path_offset_lat_accel_adjust_scale = 0.2
    self.path_offset_scale = -1.0
    self.path_offset_rate_scale = -0.07 # determined in plotjuggler to best match with `LatCtlPath_An_Actl`
    self.path_offset_rate_curvature_adjustment_bp = [17.0, 20.0] # m/s , 17 suggested by Isaac
    self.path_offset_rate_curvature_adjustment_v = [4.0, 0.0] # this adjustment turns off at higher speeds
    
    # switch from model.position.y-based path offset and path offset rate to laneline-based path offset and path offset rate
    # based on laneline confidence
    self.min_laneline_confidence_bp = [0.6, 0.8]
    
    # values from previous frame and rate limits
    self.curvature_rate_last = 0.0
    self.path_offset_last = 0.0
    self.path_offset_rate_last = 0.0

    # rate limits determined by inspecting rate limits of stock signals in plotjuggler
    self.curvature_rate_rate_limit = 6e-5
    self.path_offset_rate_limit = 0.1
    self.path_offset_rate_rate_limit = 0.015


  def update(self, CC, CS, now_nanos, model_data=None):
    can_sends = []

    actuators = CC.actuators
    hud_control = CC.hudControl

    main_on = CS.out.cruiseState.available
    steer_alert = hud_control.visualAlert in (VisualAlert.steerRequired, VisualAlert.ldw)
    fcw_alert = hud_control.visualAlert == VisualAlert.fcw

    ### acc buttons ###
    if CC.cruiseControl.cancel:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, cancel=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, cancel=True))
    elif CC.cruiseControl.resume and (self.frame % CarControllerParams.BUTTONS_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, resume=True))
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.main, CS.buttons_stock_values, resume=True))
    # if stock lane centering isn't off, send a button press to toggle it off
    # the stock system checks for steering pressed, and eventually disengages cruise control
    elif CS.acc_tja_status_stock_values["Tja_D_Stat"] != 0 and (self.frame % CarControllerParams.ACC_UI_STEP) == 0:
      can_sends.append(fordcan.create_button_msg(self.packer, self.CAN.camera, CS.buttons_stock_values, tja_toggle=True))

    ### lateral control ###
    # send steer msg at 20Hz
    if (self.frame % CarControllerParams.STEER_STEP) == 0:
      if CC.latActive:
        # apply rate limits, curvature error limit, and clip to signal range
        current_curvature = actuators.steer # used to store the current curvature, since actuators.steer isn't used for angle-control cars
        apply_curvature = apply_ford_curvature_limits(actuators.curvature, self.apply_curvature_last, current_curvature, CS.out.vEgoRaw)

        if model_data is not None and len(model_data.orientation.x) >= CONTROL_N:
          # compute curvature from model predicted lateral acceleration
          future_time = interp(CS.out.vEgo, self.future_curvature_time_bp, self.future_curvature_time_v)
          lat_accel = interp(future_time, ModelConstants.T_IDXS, model_data.acceleration.y)
          apply_curvature = lat_accel / (CS.out.vEgo ** 2)
          apply_curvature = apply_ford_curvature_limits(apply_curvature, self.apply_curvature_last, current_curvature, CS.out.vEgoRaw)

          # Computing three additional input values: desired curvature rate, path offset, and path angle (path offset rate)
          # Get desired curvature rate
          curvatures = np.array(model_data.acceleration.y) / (CS.out.vEgo ** 2)
          desired_curvature_rate = (interp(self.rate_future_time, ModelConstants.T_IDXS, curvatures) \
            - curvatures[0]) / self.rate_future_time
          
          # Get path offset and offset rate, using both model.position.y and lanelines.
          # We'll switch between the two versions using minimum laneline confidence
          lane_change = model_data.meta.laneChangeState in (LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing)
          laneline_confidence = min(model_data.laneLineProbs[1], model_data.laneLineProbs[2])
          laneline_path_offset_scale = interp(laneline_confidence, self.min_laneline_confidence_bp, [0.0, 1.0])
          # during lane change, use only model.position.y
          if lane_change:
            laneline_path_offset_scale = 0.0

          # first get path offset from model.position.y and adjust based on lat accel
          path_offset_position = interp(self.future_lookup_time, ModelConstants.T_IDXS, model_data.position.y)
          path_offset_position -= lat_accel * self.path_offset_lat_accel_adjust_scale
          # now get path offset from lanelines
          path_offset_lanelines = (model_data.laneLines[1].y[0] + model_data.laneLines[2].y[0]) / 2
          # blend the two path offsets
          path_offset = path_offset_position * (1 - laneline_path_offset_scale) + path_offset_lanelines * laneline_path_offset_scale

          # Get path offset rate
          path_offset_rate_curvature_scale = interp(CS.out.vEgo, self.path_offset_rate_curvature_adjustment_bp, self.path_offset_rate_curvature_adjustment_v)
          path_offset_rate_position = (interp(self.rate_future_time, ModelConstants.T_IDXS, model_data.position.y) \
            - model_data.position.y[0]) / self.rate_future_time
          # adjust path offset rate based on curvature
          path_offset_rate_position -= lat_accel * self.path_offset_rate_position_lat_accel_adjust_scale
          path_offset_rate_position -= apply_curvature * path_offset_rate_curvature_scale
          # now get path offset rate from lanelines
          path_offset_future = (interp(0.1, ModelConstants.T_IDXS, model_data.laneLines[1].y) + interp(0.1, ModelConstants.T_IDXS, model_data.laneLines[2].y)) / 2
          path_offset_rate_lanelines = (path_offset_future - path_offset_lanelines) / 0.1
          path_offset_rate_lanelines -= lat_accel * self.path_offset_rate_laneline_lat_accel_adjust_scale
          path_offset_rate_lanelines -= apply_curvature * path_offset_rate_curvature_scale
          self.path_offset_rate_laneline.update(path_offset_rate_lanelines)
          # blend the two path offset rates
          path_offset_rate = path_offset_rate_position * (1 - laneline_path_offset_scale) + self.path_offset_rate_laneline.x * laneline_path_offset_scale
          
          # apply scaling and rate limits
          desired_curvature_rate *= self.desired_curvature_rate_scale
          desired_curvature_rate = clip(desired_curvature_rate,
                                          self.curvature_rate_last - self.curvature_rate_rate_limit,
                                          self.curvature_rate_last + self.curvature_rate_rate_limit)
          
          path_offset *= self.path_offset_scale
          path_offset = clip(path_offset, 
                                      self.path_offset_last - self.path_offset_rate_limit, 
                                      self.path_offset_last + self.path_offset_rate_limit)
          
          path_offset_rate *= self.path_offset_rate_scale
          path_offset_rate = clip(path_offset_rate,
                                            self.path_offset_rate_last - self.path_offset_rate_rate_limit,
                                            self.path_offset_rate_last + self.path_offset_rate_rate_limit)

          # save values for next frame
          self.curvature_rate_last = desired_curvature_rate
          self.path_offset_last = path_offset
          self.path_offset_rate_last = path_offset_rate
        else:
          lane_change = False
          desired_curvature_rate = 0.0
          path_offset = 0.0
          path_offset_rate = 0.0
          self.curvature_rate_last = desired_curvature_rate
          self.path_offset_last = path_offset
          self.path_offset_rate_last = path_offset_rate
      else:
        apply_curvature = 0.
        desired_curvature_rate = 0.0
        path_offset = 0.0
        path_offset_rate = 0.0
        lane_change = False

      # trap the previous curvatures for analysis
      apply_curvature_previous2 = apply_curvature_previous
      apply_curvature_previous = apply_curvature
      self.apply_curvature_last = apply_curvature

      # Determine if we reached a curve apex (curvature is shrinking 2 scans in a row) and we were in a big curve
      if (abs(apply_curvature_previous2) > abs(apply_curvature_previous) > abs(apply_curvature)) and abs(apply_curvature_previous2) > 0.012:
          apex_reached = 1
      else:
          apex_reached = 0

      # If not in testing mode, revert back to a more stock logic
      # To-do this can be removed
      # if not self.testing_active:
        # path_offset = 0
        # path_offset_rate = 0
        # desired_curvature_rate = 0
        # lane_change = False

      # Determine if a human is making a turn
      steeringPressed = CS.out.steeringPressed
      steeringAngleDeg = CS.out.steeringAngleDeg

      if steeringPressed and (abs(steeringAngleDeg) > 60):
          human_turn = 1
      else:
          human_turn = 0

      #Determin when to reset steering
      if human_turn or apex_reached:
          reset_steering = 1
      else:
          reset_steering = 0

      # reset sterring by setting all values to 0 and ramp_type to immediate
      if reset_steering == 1:
          apply_curvature = 0
          path_offset = 0
          path_offset_rate = 0
          desired_curvature_rate = 0
          ramp_type = 3
      else:
          # Call the function here to set ramp_type based on lat_accel
          lat_accel = model_data.acceleration.y[3]
          ramp_type = set_ramp_type(lat_accel)

      if self.CP.carFingerprint in CANFD_CAR:
        # TODO: extended mode
        mode = 1 if CC.latActive else 0
        counter = (self.frame // CarControllerParams.STEER_STEP) % 0x10
        can_sends.append(fordcan.create_lat_ctl2_msg(self.packer, self.CAN, mode, 
                                                     ramp_type,
                                                     path_offset, 
                                                     path_offset_rate, 
                                                     -apply_curvature, 
                                                     desired_curvature_rate, 
                                                     counter,
                                                     lane_change))
      else:
        can_sends.append(fordcan.create_lat_ctl_msg(self.packer, self.CAN, CC.latActive, 
                                                    ramp_type,
                                                    path_offset, 
                                                    path_offset_rate, 
                                                    -apply_curvature, 
                                                    desired_curvature_rate))

    # send lka msg at 33Hz
    if (self.frame % CarControllerParams.LKA_STEP) == 0:
      can_sends.append(fordcan.create_lka_msg(self.packer, self.CAN))

    ### longitudinal control ###
    # send acc msg at 50Hz
    if self.CP.openpilotLongitudinalControl and (self.frame % CarControllerParams.ACC_CONTROL_STEP) == 0:
      # Both gas and accel are in m/s^2, accel is used solely for braking
      accel = clip(actuators.accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX)
      gas = accel
      if not CC.longActive or gas < CarControllerParams.MIN_GAS:
        gas = CarControllerParams.INACTIVE_GAS
      stopping = CC.actuators.longControlState == LongCtrlState.stopping

      precharge_actuate, brake_actuate = actuators_calc(self, accel)
      brake = accel
      if brake < 0 and brake_actuate:
        brake = interp(accel, [ CarControllerParams.ACCEL_MIN, self.brake_converge_at, self.brake_clip], [CarControllerParams.ACCEL_MIN, self.brake_converge_at, self.brake_0_point])

      # Calculate targetSpeed
      targetSpeed = clip(actuators.speed * self.target_speed_multiplier, 0, V_CRUISE_MAX)
      if not CC.longActive and hud_control.setSpeed:
        targetSpeed = hud_control.setSpeed

      can_sends.append(fordcan.create_acc_msg(self.packer, self.CAN, CC.longActive, gas, brake, stopping, brake_actuate, precharge_actuate, v_ego_kph=targetSpeed))

    ### ui ###
    send_ui = (self.main_on_last != main_on) or (self.lkas_enabled_last != CC.latActive) or (self.steer_alert_last != steer_alert)
    # send lkas ui msg at 1Hz or if ui state changes
    if (self.frame % CarControllerParams.LKAS_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_lkas_ui_msg(self.packer, self.CAN, main_on, CC.latActive, steer_alert, hud_control, CS.lkas_status_stock_values))
    # send acc ui msg at 5Hz or if ui state changes
    if (self.frame % CarControllerParams.ACC_UI_STEP) == 0 or send_ui:
      can_sends.append(fordcan.create_acc_ui_msg(self.packer, self.CAN, self.CP, main_on, CC.latActive,
                                         fcw_alert, CS.out.cruiseState.standstill, hud_control,
                                         CS.acc_tja_status_stock_values))

    self.main_on_last = main_on
    self.lkas_enabled_last = CC.latActive
    self.steer_alert_last = steer_alert

    new_actuators = actuators.copy()
    new_actuators.curvature = self.apply_curvature_last

    self.frame += 1
    return new_actuators, can_sends
