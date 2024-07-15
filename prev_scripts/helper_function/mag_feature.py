import math

# G = 9.81

def norm(magVec):
  squared_sum = 0.0
  for val in magVec:
    squared_sum += val*val
  return math.sqrt(squared_sum)

def calcMagFeatures(magVec, gravVec):
  B_h1 = magVec[0] * cosPitch(gravVec) + magVec[2] * sinPitch(gravVec)
  B_h2 = magVec[1] * cosRoll(gravVec) + magVec[0] * sinRoll(gravVec) * sinPitch(gravVec) - magVec[2] * cosPitch(gravVec) * sinRoll(gravVec)
  B_h = math.sqrt(B_h1 ** 2 + B_h2 ** 2)
  B_v = magVec[1] * sinRoll(gravVec) - magVec[0] * cosRoll(gravVec) * sinPitch(gravVec) + magVec[2] * cosPitch(gravVec) * cosRoll(gravVec)
  return [norm(magVec), B_h, B_v]
      
# x
def pitch(gravVec):
  return math.atan(gravVec[0] / math.sqrt(gravVec[1] * gravVec[1] + gravVec[2] * gravVec[2]))

def sinPitch(gravVec):
  return math.sin(pitch(gravVec))

def cosPitch(gravVec):
  return math.cos(pitch(gravVec))

# y
def roll(gravVec):
  return math.atan(gravVec[1] / math.sqrt(gravVec[0] * gravVec[0] + gravVec[2] * gravVec[2]))

def sinRoll(gravVec):
  return math.sin(roll(gravVec))

def cosRoll(gravVec):
  return math.cos(roll(gravVec))