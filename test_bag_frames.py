#!/usr/bin/env python3
"""Test script to browse and publish frames from the bag file."""
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time


class BagFrameTester(Node):
    def __init__(self):
        super().__init__('bag_frame_tester')
        self.timestamp_pub = self.create_publisher(String, '/labeling/current_timestamp', 10)
        self.get_logger().info('Bag Frame Tester ready')
    
    def publish_frame(self, frame_number):
        """Publish a frame number request multiple times to ensure delivery."""
        timestamp_info = {
            "scan_timestamp": "test",
            "frame_number": str(frame_number),
            "cluster_id": "test"
        }
        ts_msg = String()
        ts_msg.data = json.dumps(timestamp_info)
        
        # Publish multiple times to ensure at least one gets through
        for _ in range(5):
            self.timestamp_pub.publish(ts_msg)
            time.sleep(0.05)
        
        self.get_logger().info(f'Requested frame #{frame_number}')


def main():
    rclpy.init()
    node = BagFrameTester()
    
    print("\n=== Bag Frame Tester ===")
    print("This script publishes frame requests to test bag synchronization.")
    print("\nMake sure bag_frame_publisher.py is running in another terminal!")
    print("\nCommands:")
    print("  <number>   - Publish specific frame number")
    print("  b          - Publish frame from beginning (frame 0)")
    print("  m          - Publish frame from middle")
    print("  e          - Publish frame from end")
    print("  s          - Scan through frames (0, 10, 20, 30...)")
    print("  q          - Quit")
    print()
    
    # Give time for subscription to connect
    print("Waiting for publishers to connect...")
    time.sleep(2.0)
    
    # Spin to ensure connections are established
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    
    print("Ready! Enter commands.\n")
    
    try:
        while True:
            cmd = input("Enter command: ").strip()
            
            if cmd == 'q':
                break
            elif cmd == 'b':
                node.publish_frame(0)
                print("→ Published frame 0 (beginning)")
            elif cmd == 'm':
                # Estimate middle - adjust based on your bag
                node.publish_frame(500)
                print("→ Published frame 500 (middle estimate)")
            elif cmd == 'e':
                # Estimate end - adjust based on your bag
                node.publish_frame(1000)
                print("→ Published frame 1000 (end estimate)")
            elif cmd == 's':
                print("Scanning frames 0, 10, 20, 30, 40, 50...")
                for i in range(0, 60, 10):
                    node.publish_frame(i)
                    time.sleep(0.5)
                    rclpy.spin_once(node, timeout_sec=0.1)
                print("→ Scan complete")
            elif cmd.isdigit():
                frame_num = int(cmd)
                node.publish_frame(frame_num)
                print(f"→ Published frame {frame_num}")
            else:
                print("Invalid command. Try: <number>, b, m, e, s, or q")
            
            # Process ROS callbacks
            rclpy.spin_once(node, timeout_sec=0.1)
    
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("\nTester stopped.")


if __name__ == '__main__':
    main()
