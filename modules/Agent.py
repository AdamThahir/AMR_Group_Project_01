import rospy
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from modules.KalmanFilter import KalmanFilter

class Agent:
    def __init__(self, nodeName='amr', queueSize=10):
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

        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_z = msg.pose.pose.position.z

        pass

    def findDistanceBearing(self, msg):
        """
        Takes callback data as input and returns the distance and
        bearing angles to the cylindrical object
        """

        ObjectDistanceBearing = {} # Key: bearing. Value: Distance at given angle.
        for i in range(360):
            sight = msg.ranges[i]
            if not sight == float('inf'):
                bearing = 360 - i # bearing is clockwise, angle here is anticlockwise. 
                ObjectDistanceBearing[bearing] = sight

        # print (ObjectDistanceBearing)
        self.PrintObjectDistanceBearing(ObjectDistanceBearing)

        # [MOVE] move the bot
        self.move_spiral()

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
