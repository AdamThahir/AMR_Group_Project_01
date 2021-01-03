filepath = fullfile('D:\adamthahir\workspace\AMR Project\2021-01-03-23-29-01.bag');
bag = rosbag(filepath)
bagselect_odom = select(bag, 'Topic', '/odom')
pred_sub = select(bag,'Topic','/pred')
msgs_odom = readMessages(bagselect_odom);
ts_odom = timeseries(bagselect_odom, 'Pose.Pose.Position.X');
ts2_odom = timeseries(bagselect_odom,'Pose.Pose.Position.Y');
ts_odom_combined = timeseries(pred_sub, 'Pose.Pose.Position.X');
ts2_odom_combined = timeseries(pred_sub,'Pose.Pose.Position.Y');
a = ts_odom.data;
b = ts2_odom.data;
c = ts_odom_combined.data;
d = ts2_odom_combined.data;
plot(a,b,'b')
hold on;
plot(c,d,'r');
axis equal;

