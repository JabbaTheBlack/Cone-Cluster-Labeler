#!/usr/bin/env python3
"""
Script to find the closest image frame from a ROS2 bag based on a cluster timestamp.
Searches for images under the topic: /my_camera/pylon_ros2_camera_node/image_raw
"""

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


# ============================================================================
# CONFIGURATION: Hardcoded bag path
# Set your default bag file path here
# ============================================================================
script_dir = Path(__file__).parent
DEFAULT_BAG_PATH = script_dir / 'Dataset' / 'raw' / 'Skidpad' / 'fireup_11_28_29_MANUAL_0.mcap'

class ClusterImageFinder:
    def __init__(self, bag_path):
        """
        Initialize the ClusterImageFinder.
        
        Args:
            bag_path: Path to the ROS2 bag file (.mcap or .db3)
        """
        self.bag_path = Path(bag_path)
        self.image_topic = '/my_camera/pylon_ros2_camera_node/image_raw'
        
        # Cache for image messages indexed by timestamp
        self.images_by_timestamp = {}  # timestamp (microseconds) -> Image message
        self.cache_loaded = False
        
        self.bridge = CvBridge()
        
        print(f'Initialized ClusterImageFinder for bag: {self.bag_path}')
    
    def load_bag_cache(self):
        """Load all image messages from bag indexed by timestamp."""
        if self.cache_loaded:
            return
        
        print(f'Loading bag: {self.bag_path}')
        
        if not self.bag_path.exists():
            raise FileNotFoundError(f'Bag file not found: {self.bag_path}')
        
        try:
            with open(self.bag_path, 'rb') as f:
                reader = make_reader(f, decoder_factories=[DecoderFactory()])
                
                for schema, channel, message, decoded in reader.iter_decoded_messages():
                    # Filter for the specific image topic
                    if channel.topic == self.image_topic and schema.name == 'sensor_msgs/msg/Image':
                        # Get header timestamp in microseconds
                        # Format: sec * 1000000 + nsec // 1000
                        header_ts = decoded.header.stamp.sec * 1000000 + decoded.header.stamp.nanosec // 1000
                        
                        # Deserialize raw message data to proper ROS2 Image type
                        ros_msg = deserialize_message(message.data, Image)
                        self.images_by_timestamp[header_ts] = ros_msg
                
                print(f'Loaded {len(self.images_by_timestamp)} images from bag (indexed by timestamp)')
                self.cache_loaded = True
        
        except Exception as e:
            print(f'Failed to load bag: {e}')
            raise
    
    def find_closest_image(self, target_timestamp):
        """
        Find the image frame with the closest timestamp to the target.
        
        Args:
            target_timestamp: Target timestamp in microseconds (int)
        
        Returns:
            tuple: (Image message, matched_timestamp, time_difference_us)
                   or (None, None, None) if no images found
        """
        # Ensure bag is loaded
        if not self.cache_loaded:
            self.load_bag_cache()
        
        if not self.images_by_timestamp:
            print('No images found in bag')
            return None, None, None
        
        # First try exact match
        if target_timestamp in self.images_by_timestamp:
            return self.images_by_timestamp[target_timestamp], target_timestamp, 0
        
        # Find closest timestamp
        closest_ts = min(self.images_by_timestamp.keys(), key=lambda ts: abs(ts - target_timestamp))
        diff = abs(closest_ts - target_timestamp)
        
        return self.images_by_timestamp[closest_ts], closest_ts, diff
    
    def get_image_cv(self, target_timestamp):
        """
        Get the closest image as OpenCV format (numpy array).
        
        Args:
            target_timestamp: Target timestamp in microseconds (int)
        
        Returns:
            tuple: (cv_image, matched_timestamp, time_difference_us)
                   or (None, None, None) if no images found
        """
        image_msg, matched_ts, diff = self.find_closest_image(target_timestamp)
        
        if image_msg is None:
            return None, None, None
        
        # Convert ROS Image message to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            return cv_image, matched_ts, diff
        except Exception as e:
            print(f'Error converting image: {e}')
            return None, None, None
    
    def save_image_for_cluster(self, target_timestamp, output_path=None):
        """
        Find and save the closest image for a given cluster timestamp.
        
        Args:
            target_timestamp: Target timestamp in microseconds (int)
            output_path: Path to save the image (optional)
        
        Returns:
            tuple: (output_path, matched_timestamp, time_difference_us)
        """
        cv_image, matched_ts, diff = self.get_image_cv(target_timestamp)
        
        if cv_image is None:
            print(f'No image found for timestamp {target_timestamp}')
            return None, None, None
        
        # Generate output path if not provided
        if output_path is None:
            output_path = f'cluster_image_{target_timestamp}.png'
        
        # Save image
        cv2.imwrite(str(output_path), cv_image)
        
        if diff == 0:
            print(f'Saved image (exact match): {output_path}')
        else:
            print(f'Saved image (closest match, diff={diff}µs, {diff/1000:.2f}ms): {output_path}')
        
        return output_path, matched_ts, diff
    
    def get_available_timestamps(self):
        """
        Get all available image timestamps from the bag.
        
        Returns:
            list: Sorted list of timestamps in microseconds
        """
        if not self.cache_loaded:
            self.load_bag_cache()
        
        return sorted(self.images_by_timestamp.keys())


class ClusterImagePublisher(Node):
    """ROS2 node that publishes images from bag based on incoming timestamps."""

    def __init__(self, bag_path, publish_topic='/labeling/synced_image'):
        super().__init__('cluster_image_publisher')

        self.finder = ClusterImageFinder(bag_path)
        self.publish_topic = publish_topic
        self.publisher = self.create_publisher(Image, self.publish_topic, 10)
        self.subscription = self.create_subscription(
            String,
            '/labeling/current_timestamp',
            self.timestamp_callback,
            10,
        )

        # Keep the last published frame to republish so Foxglove can display it steadily.
        self.current_image = None
        self.timer = self.create_timer(0.2, self.timer_callback)  # 5 Hz

        self.get_logger().info(
            f'ClusterImagePublisher ready. Bag={bag_path}, output={self.publish_topic}'
        )

    def timer_callback(self):
        if self.current_image is not None:
            self.publisher.publish(self.current_image)

    def timestamp_callback(self, msg):
        try:
            data = json.loads(msg.data)
            scan_ts = data.get('scan_timestamp')
        except Exception as exc:
            self.get_logger().warn(f'Invalid timestamp payload: {exc}')
            return

        if scan_ts is None:
            self.get_logger().warn('timestamp payload missing scan_timestamp')
            return

        try:
            target_ts = int(scan_ts)
        except Exception:
            self.get_logger().warn(f'Invalid scan_timestamp value: {scan_ts}')
            return

        try:
            image_msg, matched_ts, diff = self.finder.find_closest_image(target_ts)
        except Exception as exc:
            self.get_logger().error(f'Lookup failed: {exc}')
            return

        if image_msg is None:
            self.get_logger().warn(f'No image found for timestamp {target_ts}')
            return

        self.current_image = image_msg
        self.publisher.publish(image_msg)

        if diff == 0:
            self.get_logger().info(f'Published image exact match ts={target_ts}')
        else:
            self.get_logger().info(
                f'Published closest image diff={diff}us ({diff/1000:.2f}ms)'
            )


def main():
    """Example usage of ClusterImageFinder."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find and extract image frames from ROS2 bag based on cluster timestamps'
    )
    parser.add_argument('--bag-path', type=str, default=DEFAULT_BAG_PATH,
                       help=f'Path to ROS2 bag file (.mcap). Default: {DEFAULT_BAG_PATH}')
    parser.add_argument('--timestamp', type=int, help='Cluster timestamp in microseconds (CLI mode)')
    parser.add_argument('--output', type=str, help='Output path for the image (CLI mode)')
    parser.add_argument('--list-timestamps', action='store_true', 
                       help='List all available image timestamps (CLI mode)')
    parser.add_argument('--publish-topic', type=str, default='/labeling/synced_image',
                        help='Image topic to publish for Foxglove (default: /labeling/synced_image)')
    
    args = parser.parse_args()
    
    # Use the bag path from arguments or default
    bag_path = args.bag_path
    
    # Check if any CLI mode flag is set
    cli_mode = args.timestamp or args.list_timestamps
    
    if not cli_mode:
        # Default: Run as ROS2 publisher node for Foxglove
        rclpy.init(args=None)
        node = ClusterImagePublisher(bag_path, args.publish_topic)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()
        return

    # Initialize finder for CLI modes
    finder = ClusterImageFinder(bag_path)
    
    if args.list_timestamps:
        # List all available timestamps
        timestamps = finder.get_available_timestamps()
        print(f'\nFound {len(timestamps)} image timestamps:')
        for i, ts in enumerate(timestamps[:10]):  # Show first 10
            print(f'  {ts}')
        if len(timestamps) > 10:
            print(f'  ... and {len(timestamps) - 10} more')
        return
    
    if args.timestamp:
        # Find and save image for specific timestamp
        finder.save_image_for_cluster(args.timestamp, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
