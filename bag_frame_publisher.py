#!/usr/bin/env python3
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from pathlib import Path
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from rclpy.serialization import deserialize_message


class BagFramePublisher(Node):
    def __init__(self):
        super().__init__('bag_frame_publisher')
        
        # Subscribe to timestamp requests from cluster_labeler
        self.subscription = self.create_subscription(
            String,
            '/labeling/current_timestamp',
            self.timestamp_callback,
            10
        )
        
        # Publish synced frames from bag
        self.publisher = self.create_publisher(PointCloud2, '/labeling/synced_frame', 10)
        
        # Path to mcap bag
        self.bag_path = Path('/home/praksz/FRT2026/Cone-Labeler/Cone-Cluster-Labeler/Dataset/skidpad_transformed.mcap')
        
        # Cache for bag messages indexed by timestamp
        self.frames_by_timestamp = {}  # timestamp -> PointCloud2 message
        self.cache_loaded = False
        self.current_frame = None  # Currently selected frame to publish continuously
        
        # Timer to republish current frame
        self.timer = self.create_timer(0.20, self.timer_callback)  # 5 Hz
        
        self.get_logger().info('Bag Frame Publisher ready')
    
    def load_bag_cache(self):
        """Load all pointcloud messages from bag indexed by timestamp."""
        if self.cache_loaded:
            return
        
        self.get_logger().info(f'Loading bag: {self.bag_path}')
        
        try:
            with open(self.bag_path, 'rb') as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                
                for schema, channel, message, decoded in reader.iter_decoded_messages():
                    # Collect PointCloud2 messages indexed by header timestamp
                    if schema.name == 'sensor_msgs/msg/PointCloud2':
                        # Get header timestamp in the same format as cluster filenames
                        # Format: sec * 1000000 + nsec // 1000 (microseconds)
                        header_ts = decoded.header.stamp.sec * 1000000 + decoded.header.stamp.nanosec // 1000
                        
                        # Deserialize raw message data to proper ROS2 type for publishing
                        ros_msg = deserialize_message(message.data, PointCloud2)
                        self.frames_by_timestamp[header_ts] = ros_msg
                
                self.get_logger().info(f'Loaded {len(self.frames_by_timestamp)} frames from bag (indexed by timestamp)')
                self.cache_loaded = True
        
        except Exception as e:
            self.get_logger().error(f'Failed to load bag: {e}')
    
    def timer_callback(self):
        """Continuously publish the current frame."""
        if self.current_frame is not None:
            self.publisher.publish(self.current_frame)
    
    def find_closest_frame(self, target_ts):
        """Find the frame with the closest timestamp."""
        if not self.frames_by_timestamp:
            return None
        
        # First try exact match
        if target_ts in self.frames_by_timestamp:
            return self.frames_by_timestamp[target_ts], target_ts, 0
        
        # Find closest timestamp
        closest_ts = min(self.frames_by_timestamp.keys(), key=lambda ts: abs(ts - target_ts))
        diff = abs(closest_ts - target_ts)
        return self.frames_by_timestamp[closest_ts], closest_ts, diff
    
    def timestamp_callback(self, msg):
        """Receive timestamp info and publish matching frame by timestamp."""
        try:
            timestamp_info = json.loads(msg.data)
            scan_timestamp = timestamp_info.get('scan_timestamp')
            
            if not scan_timestamp:
                self.get_logger().warn('No scan_timestamp in timestamp info')
                return
            
            # Load bag cache on first request
            if not self.cache_loaded:
                self.load_bag_cache()
            
            # Convert scan timestamp to integer
            try:
                target_ts = int(scan_timestamp)
            except:
                self.get_logger().warn(f'Invalid scan_timestamp: {scan_timestamp}')
                return
            
            # Find matching frame by timestamp
            result = self.find_closest_frame(target_ts)
            if result:
                frame_msg, matched_ts, diff = result
                self.current_frame = frame_msg  # Store for continuous publishing
                self.publisher.publish(frame_msg)
                
                if diff == 0:
                    self.get_logger().info(f'Published frame (exact match, ts={target_ts})')
                else:
                    self.get_logger().info(f'Published frame (closest match, diff={diff}µs)')
            else:
                self.get_logger().warn(f'No frame found for timestamp {target_ts}')
        
        except Exception as e:
            self.get_logger().error(f'Error processing timestamp: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = BagFramePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
