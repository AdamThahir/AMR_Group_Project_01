import rospy
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from modules.KalmanFilter import KalmanFilter, EKF_SLAM
from modules.ParticleFilter import ParticleFilter
import numpy as np
from numpy.random import randn
import scipy.stats
import time

class Agent:
    def __init__(self, filter_type = 'kalman', nodeName='amr', queueSize=10):
        """
        Initiation for the bot. Setting defaults and adding Subscribers/
        Publishers. 
        """
        self.object_detected = False

        self.SetSpeed(0.5, 0, 0)
        self.current_x = None
        self.current_y = None
        self.current_z = None
        
        self.DEFAULT_SPIRAL_RADIUS = 10
        self.spiral_radius = self.DEFAULT_SPIRAL_RADIUS
        self.print_message = False
        self.filter_type = filter_type

        self.move_switch = False
        self.timer_start = time.time()

        if ('kalman' in filter_type):
        	self.filter = KalmanFilter()
        elif ('slam' in filter_type):
            # nObjects is True. Others I am guessing based on the environment. Should get back on it.
            # R i am really not sure about as I got it from the cited notebook.
            nObjects = 50
            initialPosition = np.array([3, 3, np.pi/2])
            self.R = np.array([[.001, .0, .0],
              [.0, .001, .0],
              [.0, .0, .0001]])
            self.X_hat = []
            self.Conv_hat = []
            self.positions = []
            self.filter = EKF_SLAM(initialPosition, nObjects, self.R)

            self.step_count = 0
        else :
        	self.filter = ParticleFilter()

        rospy.init_node(nodeName)
        
        # Subscribers
        self.laserSubscriber = rospy.Subscriber('/scan', LaserScan, self.LaserCallback)
        self.odometerySubscriber = rospy.Subscriber('odom', Odometry, self.OdometryCallback)

        # Publisher
        self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size=queueSize)


        self.move = Twist()

        # Loop -- I think.
        rospy.spin()

    def SetSpeedX(self, speed):
        """Set X Speed"""
        self.x_speed = speed
    
    def SetSpeedY(self, speed):
        """Set Y Speed"""
        self.y_speed = speed
    
    def SetSpeedZ(self, speed):
        """Set Z Speed"""
        self.z_speed = speed

    def SetSpeed(self, speedX, speedY, speedZ):
        """Set X, Y and Z Speeds"""
        self.SetSpeedX(speedX)
        self.SetSpeedY(speedY)
        self.SetSpeedZ(speedZ)

    def LaserCallback(self, msg):
        """Callback for the Laser Sensor"""
        self.findDistanceBearing(msg)
        pass

    def OdometryCallback(self, msg):
        """Callback for Odometry messages"""

        print (len(self.positions))
        print (msg.pose.pose)
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z
        self.positions.append([self.current_x, self.current_y, self.current_z])

        if (self.filter_type == 'kalman'):

            # print (self.current_x, self.current_y, self.current_z)
            Z = np.matrix([self.current_x, self.current_y,self.current_z]).T

            self.filter.update(Z)
            filter_X = self.filter.predict()

            with open('res_measured_kalman.csv', 'a') as f:
                f.write(f'{self.current_x};{self.current_y};{self.current_z}\n')

            with open('res_predicted_kalman.csv', 'a') as f:
                f.write(f'{filter_X[0]};{filter_X[1]};{filter_X[2]}\n')
        
        elif (self.filter_type == 'slam'):

            with open('slam_res_measured_part.csv', 'a') as f:
                f.write(f'{self.current_x};{self.current_y}\n')

            # print (len(self.positions))
            # 101 TODO
            # we calculate after N positions
            if (len(self.positions) % 501 == 0):
                U = self.filter.get_U()

                for t, u in enumerate(U):
                    z = self.filter.get_Z(self.positions[t])
                    x_hat, Cov = self.filter.filter(z,u)
                    self.X_hat.append(x_hat)
                    self.Conv_hat.append(Cov)
                
                with open('x.csv', 'w') as f:
                    for i in range(len(self.positions)):
                        val = ",".join(str(x) for x in self.positions[i]) + ";"
                        val += ",".join(str(x) for x in self.X_hat[i][:3]) + "\r\n"
                        f.write(val)


                print ('Xhat size: ', len(self.X_hat), '\npos:', len(self.positions))

                with open('slam_res_x_hat_0_part.csv', 'a') as f:
                    f.write(f'{self.X_hat[0][0]};{self.X_hat[0][1]}\n')

                with open('slam_res_x_hat_1_part.csv', 'a') as f:
                    f.write(f'{self.X_hat[1][0]};{self.X_hat[1][1]}\n')

                # print (f'shape: {np.asarray(X_hat).shape}')

                X_hat = self.X_hat
                Conv_hat = self.Conv_hat
                avg = np.zeros((len(X_hat), 2))
                for i in range(len(X_hat)):
                    avg[i][0] += X_hat[i][0]
                    avg[i][1] += X_hat[i][1]

                for i in range(len(X_hat)):
                    avg[i][0] /= len(X_hat)
                    avg[i][1] /= len(X_hat)

                with open('slam_res_x_hat_avg_part.csv', 'a') as f:
                    f.write(f'{avg[0][0]};{avg[0][1]}\n')

                with open('slam_res_conv_part.csv', 'a') as f:
                    f.write(f'{Conv_hat}\n')

            
        else :          
            self.filter.predict([self.x_speed, (self.x_speed/self.spiral_radius)])
            self.filter.update([self.current_x, self.current_y])

            # if there are not much effieicnt particles
            if(self.filter.neff() < self.filter.N/2):
                inds = self.filter.systematic_resample()
                self.filter.resample_from_index(inds)

            mu, var = self.filter.estimate()
            with open('res_pred_part.csv', 'a') as f:
                f.write(f'{mu[0]};{mu[1]}\n')

            with open('res_measured_part.csv', 'a') as f:
                f.write(f'{self.current_x};{self.current_y}\n')



    def findDistanceBearing(self, msg):
        """
        Takes callback data as input and returns the distance and
        bearing angles to the cylindrical object
        """

        ObjectDistanceBearing = {} # Key: bearing. Value: Distance at given angle.

        # we want a total of 120 degrees. I split it 60 clockwise, 60 anticlockwise
        for i in range(60): # clockwise 60
            sight = msg.ranges[i]
            if not sight == float('inf'):
                bearing = 360 - i # bearing is clockwise, angle here is anticlockwise. 
                ObjectDistanceBearing[bearing] = sight

            if (i <= 30 and sight < 0.6):
                self.x_speed *= -1
            elif self.x_speed < 0:
                self.x_speed *= -1




        for i in range(300, 360): # anticlockwise 60
            sight = msg.ranges[i]
            if not sight == float('inf'):
                bearing = 360 - i # bearing is clockwise, angle here is anticlockwise.
                ObjectDistanceBearing[bearing] = sight

            if (i >= 330 and sight < 0.6):
                self.x_speed *= -1
            elif self.x_speed < 0:
                self.x_speed *= -1



        # print (ObjectDistanceBearing)
        self.PrintObjectDistanceBearing(ObjectDistanceBearing)

        # [MOVE] move the bot
        if not self.move_switch:
            self.move_spiral()
        else:
            self.move_line()

        return ObjectDistanceBearing

    def move_spiral(self):
        """
        Move the Bot in an oval kind of path.
        """
        self.move.linear.x = self.x_speed
        self.move.angular.z = self.x_speed/self.spiral_radius

        if self.spiral_radius >= 1:
            self.spiral_radius -= self.spiral_radius * 0.10
        else:
            self.spiral_radius = self.DEFAULT_SPIRAL_RADIUS * 0.85
        # print (self.spiral_radius)
        self.publisher.publish(self.move)

        # if (time.time() - self.timer_start) >= 10:
        #     self.move_switch = True
        #     self.timer_start = time.time()


    def move_line(self):
        """
        Move the Bot in an line kind of path.
        """
        self.move.linear.x = self.x_speed

        self.publisher.publish(self.move)

        if (time.time() - self.timer_start) >= 10:
            self.move_switch = False
            self.timer_start = time.time()

    def PrintObjectDistanceBearing(self, objectDistanceBearing, nearestN=10):
        if not self.print_message:
            self.print_message = True
            print (f'bearings are printed only once to the nearest {nearestN}')
            # print (f'distance is the first distance in the range\n')

        roundFunction = lambda x: nearestN * round(x/nearestN)
        printedKeys = []


        for key, value in objectDistanceBearing.items():
            roundKey = roundFunction(key)
            if not roundKey in printedKeys:
                printedKeys.append(roundKey)
                print (f'[{key}]: {value}')
                
        if len(printedKeys) > 0:
            print ('')
