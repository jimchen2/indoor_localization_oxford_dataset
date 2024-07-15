import json
import os
import sys
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper_function.mag_feature import calcMagFeatures

class DataPreprocessor:
    """
    mag_feature = [[Bv, Bh, Bp]]    in desired_frequency
    gt = [[x, y]]                   in desired_frequency
    """
    def __init__(self, path_folder_path, mag_period=10, grav_period=5, desired_frequency=50):
        MAG_SENSOR_NUM = 1
        GRAV_SENSOR_NUM = 5
        self.path_folder_path = path_folder_path
        files = os.listdir(path_folder_path)
        session_names = [ f for f in files if not os.path.isfile(os.path.join(path_folder_path, f)) ]
        self.path_coor = []
        ratio_array = [0]
        with open(os.path.join(path_folder_path, "shape_description.shape"), "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                segments = line.split("  ")
                self.path_coor.append([ float(segments[0]), float(segments[1]) ])
        sum_dist = 0
        for idx, pt in enumerate(self.path_coor[1:]):
            sum_dist += self.dist(self.path_coor[idx-1],self.path_coor[idx])
            ratio_array.append(sum_dist)
        self.ratio_array = [ r/sum_dist for r in ratio_array ]

        self.mag_feature = []
        self.gt = []
        self.sess_name = []
        for session in session_names:
            mag_data = [] # [ts, x, y, z]
            grav_data = [] # [ts, x, y, z]
            if not os.path.exists(os.path.join(path_folder_path, session, f"{MAG_SENSOR_NUM}.dat")):
                continue
            with open(os.path.join(path_folder_path, session, f"{MAG_SENSOR_NUM}.dat"), "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    segments = line.split("\t")
                    ts = float(segments[1])
                    segments = json.loads(segments[2])
                    segments = segments["vals"]
                    mag_data.append([ ts, float(segments[0]), float(segments[1]), float(segments[2]) ])

        
            with open(os.path.join(path_folder_path, session, f"{GRAV_SENSOR_NUM}.dat"), "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[1:]:
                    segments = line.split("\t")
                    ts = float(segments[1])
                    segments = json.loads(segments[2])
                    segments = segments["vals"]
                    grav_data.append([ ts, float(segments[0]), float(segments[1]), float(segments[2]) ])

            sessoin_inverted = False
            with open(os.path.join(path_folder_path, session, "sess_config.json"), 'r') as file:
                # Load the JSON data
                data = json.load(file)
                if data["startVertexId"] != 0:
                    sessoin_inverted = True

            mag_feature_in_sess = []
            gt_in_sess = []
            num_processed_data = int(len(mag_data)/(1000/desired_frequency/mag_period))
            process_period = int(1000/desired_frequency/mag_period)
            for processed_data_idx in range(num_processed_data):
                starting_data_idx = processed_data_idx*process_period
                ending_data_idx = (processed_data_idx+1)*process_period # not inclusive
                mag = mag_data[starting_data_idx:ending_data_idx]
                start_ts = mag[0][0]
                end_ts = mag[-1][0]
                grav = [ g[1:] for g in grav_data if g[0]>=start_ts and g[0]<end_ts ]
                if len(grav)==0:
                    continue
                mag_x = [ m[1] for m in mag ]
                mag_y = [ m[2] for m in mag ]
                mag_z = [ m[3] for m in mag ]
                PAA_mag = [ sum(mag_x)/len(mag_x), sum(mag_y)/len(mag_y), sum(mag_z)/len(mag_z) ]
                grav_x = [ g[0] for g in grav ]
                grav_y = [ g[1] for g in grav ]
                grav_z = [ g[2] for g in grav ]
                PAA_grav = [ sum(grav_x)/len(grav_x), sum(grav_y)/len(grav_y), sum(grav_z)/len(grav_z) ]
                mag_feature_in_sess.append(self.calMagFeature(PAA_mag, PAA_grav))
                if not sessoin_inverted:
                    gt_in_sess.append(self.coor_at_ratio((2*processed_data_idx+1)/2/num_processed_data))
                else:
                    gt_in_sess.append(self.coor_at_ratio(1-(2*processed_data_idx+1)/2/num_processed_data))
            self.mag_feature.append(mag_feature_in_sess)
            self.gt.append(gt_in_sess)
            self.sess_name.append(session)
    
    def dist(self, p1, p2):
        return math.sqrt(math.pow(p1[0]-p2[0],2)+math.pow(p1[1]-p2[1],2))
    
    def coor_at_ratio(self, target_ratio):
        #print(f"target_ratio = {target_ratio}")
        for idx, ratio in enumerate(self.ratio_array):
            if ratio>target_ratio:
                ratio_at_segment = (target_ratio-self.ratio_array[idx-1]) / (self.ratio_array[idx]-self.ratio_array[idx-1])
                starting_pt = self.path_coor[idx-1]
                ending_pt = self.path_coor[idx]
                diff_verctor = [ending_pt[0]-starting_pt[0], ending_pt[1]-starting_pt[1]]
                #print(f"self.path_coor = {self.path_coor}, idx = {idx}, ratio = {ratio}")
                #print(f"starting_pt = {starting_pt}, ending_pt = {ending_pt}, diff_verctor = {diff_verctor}, ratio_at_segment = {ratio_at_segment}")
                #print(f"result = {[ starting_pt[0]+ratio_at_segment*diff_verctor[0], starting_pt[1]+ratio_at_segment*diff_verctor[1] ]}")
                return [ starting_pt[0]+ratio_at_segment*diff_verctor[0], starting_pt[1]+ratio_at_segment*diff_verctor[1] ]
    
    def calMagFeature(self, mag, grav):
        magnitude = math.sqrt(sum(component**2 for component in grav))
        grav_norm = [component / magnitude for component in grav]
        dot_product = sum(component1 * component2 for component1, component2 in zip(mag, grav_norm))
        mag_along_grav = [component * dot_product for component in grav_norm]
        mag_orth_grav = [component1 - component2 for component1, component2 in zip(mag, mag_along_grav)]
        magnitide_along_grav = math.sqrt(sum(component**2 for component in mag_along_grav))
        if dot_product<0: 
            magnitide_along_grav = -magnitide_along_grav
        magnitide_orth_grav = math.sqrt(sum(component**2 for component in mag_orth_grav))
        return [magnitide_along_grav,magnitide_orth_grav,math.sqrt(sum(component**2 for component in mag))]