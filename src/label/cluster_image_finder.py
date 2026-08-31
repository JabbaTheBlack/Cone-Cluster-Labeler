#!/usr/bin/env python3
"""
Find and stream the camera image closest to a lidar cluster's scan.

The cluster .pcd filenames encode the pointcloud header stamp (lidar uptime
clock), while the camera images carry epoch header stamps — the two clocks are
unrelated, so header stamps cannot be compared across sensors. Instead we
bridge through the MCAP log_time (the common recording clock):

    cluster ts (= pointcloud header stamp, µs)
        -> pointcloud log_time
        -> image with the closest log_time

Images are NOT cached (a bag holds 3-5 GB of raw frames). Startup builds a
lightweight timestamp index in one pass, and each lookup seeks into the bag to
decode just the one image needed.

Run without arguments to start the ROS2 publisher node: it listens on
/labeling/current_timestamp (published by label/label.py) and streams the
matching image on /labeling/synced_image for Foxglove.
"""

import bisect
import json
import struct
from pathlib import Path

import rclpy


def find_project_root(start_path: Path | None = None) -> Path:
    """Walk upward until we find the repository root containing Dataset and src."""
    current = (start_path or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if (candidate / "Dataset").exists() and (candidate / "src").exists():
            return candidate

    return current


REPO_ROOT = find_project_root()
from rclpy.node import Node
from std_msgs.msg import String
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


# ============================================================================
# CONFIGURATION: Hardcoded bag path
# Keep this pointing at the same bag label.py is labeling clusters from.
# ============================================================================
DEFAULT_BAG_PATH = Path('/home/jabba/Downloads/fireup_11_28_29_MANUAL_0.mcap')

DEFAULT_IMAGE_TOPIC = '/my_camera/pylon_ros2_camera_node/image_raw'
DEFAULT_POINTCLOUD_TOPIC = '/ouster/points'


class ClusterImageFinder:
    def __init__(self, bag_path, image_topic=DEFAULT_IMAGE_TOPIC,
                 pointcloud_topic=DEFAULT_POINTCLOUD_TOPIC):
        """
        Args:
            bag_path: Path to the ROS2 bag file (.mcap)
            image_topic: Camera topic to pull frames from
            pointcloud_topic: Lidar topic whose header stamps the cluster
                              filenames are derived from
        """
        self.bag_path = Path(bag_path)
        self.image_topic = image_topic
        self.pointcloud_topic = pointcloud_topic

        # Index built once at load time (timestamps only, no image data):
        self.scan_log_by_header = {}   # pointcloud header stamp (µs) -> log_time (ns)
        self.scan_headers = []         # sorted pointcloud header stamps (µs)
        self.image_log_times = []      # sorted image log_times (ns)
        self.index_loaded = False

        self._bag_file = None
        self._reader = None

        self.bridge = CvBridge()

        print(f'Initialized ClusterImageFinder for bag: {self.bag_path}')

    def load_index(self):
        """One streaming pass over the bag collecting timestamps only."""
        if self.index_loaded:
            return

        if not self.bag_path.exists():
            raise FileNotFoundError(f'Bag file not found: {self.bag_path}')

        print(f'Indexing bag (one pass, no image caching): {self.bag_path}')

        self._bag_file = open(self.bag_path, 'rb')
        self._reader = make_reader(self._bag_file)

        for schema, channel, message in self._reader.iter_messages(
                topics=[self.image_topic, self.pointcloud_topic]):
            if channel.topic == self.pointcloud_topic:
                # Header is the first field of the serialized message: after the
                # 4-byte CDR encapsulation come stamp.sec (int32) and
                # stamp.nanosec (uint32). Parsing them directly avoids decoding
                # the full multi-MB cloud.
                sec, nsec = struct.unpack_from('<iI', message.data, 4)
                header_us = sec * 1_000_000 + nsec // 1_000
                self.scan_log_by_header[header_us] = message.log_time
            else:
                self.image_log_times.append(message.log_time)

        self.scan_headers = sorted(self.scan_log_by_header.keys())
        self.image_log_times.sort()
        self.index_loaded = True

        print(f'Indexed {len(self.scan_headers)} scans and '
              f'{len(self.image_log_times)} images')

        if not self.image_log_times:
            print(f'WARNING: no images on topic {self.image_topic}')
        if not self.scan_headers:
            print(f'WARNING: no pointclouds on topic {self.pointcloud_topic}')

    @staticmethod
    def _closest(sorted_values, target):
        """Closest value in a sorted list."""
        i = bisect.bisect_left(sorted_values, target)
        candidates = sorted_values[max(0, i - 1):i + 1]
        return min(candidates, key=lambda v: abs(v - target))

    def _fetch_image(self, log_time):
        """Seek into the bag and decode the single image at this log_time."""
        for schema, channel, message in self._reader.iter_messages(
                topics=[self.image_topic],
                start_time=log_time, end_time=log_time + 1):
            return deserialize_message(message.data, Image)
        return None

    def find_closest_image(self, target_timestamp):
        """
        Find the image recorded closest to the scan with this header stamp.

        Args:
            target_timestamp: Cluster/scan timestamp in microseconds (the
                              pointcloud header stamp from the .pcd filename)

        Returns:
            tuple: (Image message, image_log_time_us, time_difference_us)
                   where the difference is between the image and the scan on
                   the recording clock, or (None, None, None) if not found.
        """
        if not self.index_loaded:
            self.load_index()

        if not self.image_log_times or not self.scan_headers:
            return None, None, None

        # Map cluster timestamp -> scan log_time
        scan_header = self._closest(self.scan_headers, target_timestamp)
        if scan_header != target_timestamp:
            print(f'Note: no exact scan for ts={target_timestamp}, '
                  f'using nearest scan ts={scan_header} '
                  f'({abs(scan_header - target_timestamp)}µs away)')
        scan_log = self.scan_log_by_header[scan_header]

        # Find image recorded closest to that scan
        image_log = self._closest(self.image_log_times, scan_log)
        diff_us = abs(image_log - scan_log) // 1_000

        image_msg = self._fetch_image(image_log)
        if image_msg is None:
            print(f'Failed to fetch image at log_time {image_log}')
            return None, None, None

        return image_msg, image_log // 1_000, diff_us

    def get_image_cv(self, target_timestamp):
        """
        Get the closest image as OpenCV format (numpy array).

        Returns:
            tuple: (cv_image, image_log_time_us, time_difference_us)
                   or (None, None, None) if no images found
        """
        image_msg, matched_ts, diff = self.find_closest_image(target_timestamp)

        if image_msg is None:
            return None, None, None

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
            return cv_image, matched_ts, diff
        except Exception as e:
            print(f'Error converting image: {e}')
            return None, None, None

    def save_image_for_cluster(self, target_timestamp, output_path=None):
        """
        Find and save the closest image for a given cluster timestamp.

        Returns:
            tuple: (output_path, image_log_time_us, time_difference_us)
        """
        cv_image, matched_ts, diff = self.get_image_cv(target_timestamp)

        if cv_image is None:
            print(f'No image found for timestamp {target_timestamp}')
            return None, None, None

        if output_path is None:
            output_path = f'cluster_image_{target_timestamp}.png'

        cv2.imwrite(str(output_path), cv_image)
        print(f'Saved image (camera-to-scan offset {diff}µs, {diff/1000:.2f}ms): {output_path}')

        return output_path, matched_ts, diff

    def get_available_timestamps(self):
        """
        Sorted scan timestamps (µs) that clusters can be looked up by —
        these match the scan_<ts> part of the .pcd filenames.
        """
        if not self.index_loaded:
            self.load_index()
        return list(self.scan_headers)


class ClusterImagePublisher(Node):
    """ROS2 node that streams the image matching each labeled cluster."""

    def __init__(self, bag_path, publish_topic='/labeling/synced_image'):
        super().__init__('cluster_image_publisher')

        self.finder = ClusterImageFinder(bag_path)
        self.finder.load_index()

        self.publish_topic = publish_topic
        self.publisher = self.create_publisher(Image, self.publish_topic, 10)
        self.subscription = self.create_subscription(
            String,
            '/labeling/current_timestamp',
            self.timestamp_callback,
            10,
        )

        # label.py republishes the same timestamp continuously; only hit the
        # bag when it actually changes, and keep the last frame to republish
        # so Foxglove displays it steadily.
        self.current_target = None
        self.current_image = None
        self.timer = self.create_timer(0.2, self.timer_callback)  # 5 Hz

        self.get_logger().info(
            f'ClusterImagePublisher ready. Bag={bag_path}, output={self.publish_topic}'
        )

    def timer_callback(self):
        if self.current_image is not None:
            self.current_image.header.stamp = self.get_clock().now().to_msg()
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

        if target_ts == self.current_target:
            return

        try:
            image_msg, matched_ts, diff = self.finder.find_closest_image(target_ts)
        except Exception as exc:
            self.get_logger().error(f'Lookup failed: {exc}')
            return

        if image_msg is None:
            self.get_logger().warn(f'No image found for timestamp {target_ts}')
            return

        self.current_target = target_ts
        self.current_image = image_msg
        self.current_image.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.current_image)

        self.get_logger().info(
            f'Published image for scan ts={target_ts} '
            f'(camera-to-scan offset {diff/1000:.2f}ms)'
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
                       help='List available scan timestamps (CLI mode)')
    parser.add_argument('--publish-topic', type=str, default='/labeling/synced_image',
                        help='Image topic to publish for Foxglove (default: /labeling/synced_image)')

    args = parser.parse_args()

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
        timestamps = finder.get_available_timestamps()
        print(f'\nFound {len(timestamps)} scan timestamps:')
        for ts in timestamps[:10]:
            print(f'  {ts}')
        if len(timestamps) > 10:
            print(f'  ... and {len(timestamps) - 10} more')
        return

    if args.timestamp:
        finder.save_image_for_cluster(args.timestamp, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
