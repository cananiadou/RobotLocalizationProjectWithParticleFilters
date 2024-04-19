# RobotLocalizationProject
### Files included ###

Code   : RobotLocalizationWithParticleFilters.py        # implementation of particles filters algorithm
	 SimpleBicycleMotionModel.py			# implementation of simple bicycle kinematic model
	 ReadGazeboData.py				# read gazebo log files and select data
	 CarTrackNodes.py				# annotation of car track paths
	 ReportLab.py					# create report of results



### Description ###

This is an implementation of particles filtering algorithm on calculating the position of a robot, utilizing 
traffic data gathered from gazebo simulation environment.



### How to run ###

Run the file RobotLocalizationWithParticleFilters.py with the below parameters.

Arguments 		
-particles, required=True, help=Number of particles
-car_track, required=True, 0: ramp2roundabout, 
			1: roundabout2ramp,
			2: parking2roundabout,
			3: roundabout2parking,
			4: highwayUp, 
			5: highwayDown, 
			6: bumpy,
			7: manualIntersections, 
			8: ramp2rampX3, 
			9: highway2roundabout

-discarded, default=true, help=Use discarded model
-visualize_loc, default=true, help=Visualize localization nodes
-delete_old, default=true, help=Delete old files
-zero_weights, default=true, help=Zero weights when out of lane
-extra_punishment, default=0, help=Extra punishment when updating weights
-resampling, default=0, help=Resmpling method
                    0: multinominal resampling (default),
                    1: residual resampling,
                    2: stratified resampling,
                    3: systematic resampling
								 
