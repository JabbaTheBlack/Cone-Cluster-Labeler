#!/usr/bin/env python3
import json
import sys
import select
import numpy as np
from pathlib import Path
import struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import std_msgs.msg
import sensor_msgs_py.point_cloud2 as pc2


class ClusterLabelerNode(Node):
    def __init__(self):
        super().__init__('cluster_labeler')
        self.publisher = self.create_publisher(PointCloud2, '/labeling/current_cluster', 10)
        self.timestamp_pub = self.create_publisher(std_msgs.msg.String, '/labeling/current_timestamp', 10)
        self.clusters_dir = Path('/home/praksz/FRT2026/Cone-Labeler/Cone-Cluster-Labeler/Dataset/cone_clusters')
        self.output_json = Path('/home/praksz/FRT2026/Cone-Labeler/Cone-Cluster-Labeler/Dataset/labeled_clusters.json')
        self.labels = {}
        
        if self.output_json.exists():
            with open(self.output_json) as f:
                self.labels = json.load(f)
        
        self.get_logger().info('Cluster Labeler ready')
    
    def load_pcd_binary(self, filepath):
        """Load binary PCD file."""
        with open(filepath, 'rb') as f:
            while True:
                line = f.readline().decode('utf-8').strip()
                if line.startswith('DATA'):
                    break
            points = []
            while True:
                data = f.read(16)
                if not data:
                    break
                try:
                    x, y, z, intensity = struct.unpack('ffff', data)
                    points.append([x, y, z, intensity])
                except:
                    break
        return np.array(points, dtype=np.float32) if points else None
    
    def publish_cluster(self, cloud, timestamp_info=None):
        """Publish cluster to Foxglove."""
        xyz = cloud[:, :3]
        
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='lidar')
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        
        msg = pc2.create_cloud(header, fields, cloud)
        self.publisher.publish(msg)
        
        # Publish timestamp for bag synchronization
        if timestamp_info:
            ts_msg = std_msgs.msg.String()
            ts_msg.data = json.dumps(timestamp_info)
            self.timestamp_pub.publish(ts_msg)
    
    def extract_timestamp_info(self, filename):
        """Extract scan time and frame info from filename.
        Expected format: scan_<timestamp>_frame_<frame_num>_cluster_<id>.pcd
        Returns a dict or None if not matched.
        """
        try:
            import re
            m = re.match(r"scan_([^_]+)_frame_([^_]+)_cluster_([^_]+)\.pcd", filename)
            if m:
                return {
                    "scan_timestamp": m.group(1),
                    "frame_number": m.group(2),
                    "cluster_id": m.group(3),
                }
        except Exception:
            pass
        return None

    def print_stats(self, cloud, timestamp_info=None):
        """Print cluster stats."""
        xyz = cloud[:, :3]
        intensity = cloud[:, 3]
        
        if timestamp_info:
            print(f'  Scan Time: {timestamp_info.get("scan_timestamp", "?")}')
            print(f'  Frame: {timestamp_info.get("frame_number", "?")} | Cluster: {timestamp_info.get("cluster_id", "?")}')
        print(f'  Points: {len(xyz):,}')
        print(f'  X: [{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}]')
        print(f'  Y: [{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}]')
        print(f'  Z: [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]')
        h = xyz[:,2].max() - xyz[:,2].min()
        w = xyz[:,0].max() - xyz[:,0].min()
        print(f'  Size: H={h:.2f}m W={w:.2f}m (H/W={h/w:.1f})')
        print(f'  Intensity: [{intensity.min():.3f}, {intensity.max():.3f}] mean={intensity.mean():.3f}')

    def find_resume_idx(self, cluster_files):
        """Find index after LAST labeled file chronologically."""
        print(f'Loaded {len(self.labels)} existing labels')
        
        labeled_files = sorted(self.labels.keys())
        if labeled_files:
            last_labeled_file = labeled_files[-1]
            print(f'Last labeled: {last_labeled_file}')
            
            # Find next file after last labeled
            for idx, pcd_file in enumerate(cluster_files):
                if pcd_file.name > last_labeled_file:
                    resume_idx = idx - 1  # -1 to start from CURRENT position
                    print(f'Resuming from: {pcd_file.name} (#{resume_idx+1})')
                    return resume_idx
        
        print('No labels - starting from beginning')
        return 0





    def run(self):
        """Main labeling loop."""
        cluster_files = sorted(self.clusters_dir.glob('scan_*_frame_*_cluster_*.pcd'))
        
        print(f'Found {len(cluster_files)} clusters')
        print('Controls: b=blue, Y=yellow, u=unknown, n=not cone, s=skip, q=quit\n')
        
        # RESUME FROM LAST LABELED
        idx = self.find_resume_idx(cluster_files)
        
        while idx < len(cluster_files):
            pcd_file = cluster_files[idx]
            filename = pcd_file.name
            
            # Skip if already labeled
            if filename in self.labels:
                label_info = self.labels[filename]
                cone_str = f"CONE ({label_info['color']})" if label_info['is_cone'] else "NOT CONE"
                print(f'[{idx+1}/{len(cluster_files)}] {filename} - {cone_str}')
                idx += 1
                continue
            
            print(f'\n[{idx+1}/{len(cluster_files)}] {filename}')
            
            # Extract timestamp info from filename
            timestamp_info = self.extract_timestamp_info(filename)
            
            # Load cluster
            cloud = self.load_pcd_binary(str(pcd_file))
            if cloud is None:
                print('  ERROR: Could not load PCD')
                idx += 1
                continue
            
            self.print_stats(cloud, timestamp_info)
            print('Publishing to Foxglove...')
            print('Label: (b/Y/u/n/s/q) >> ', end='', flush=True)
            
            labeled = False
            while not labeled:
                # Stream the current cluster continuously
                self.publish_cluster(cloud)
                
                # Publish timestamp continuously for synchronization
                if timestamp_info:
                    try:
                        ts_msg = std_msgs.msg.String()
                        ts_msg.data = json.dumps(timestamp_info)
                        self.timestamp_pub.publish(ts_msg)
                    except Exception:
                        pass
                
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1)
                    
                    if key in '\n\r\t ':
                        continue
                    
                    if key == 'b':
                        self.labels[filename] = {'is_cone': True, 'color': 'blue'}
                        print('✓ CONE (blue)')
                        labeled = True
                        idx += 1
                    elif key == 'Y':
                        self.labels[filename] = {'is_cone': True, 'color': 'yellow'}
                        print('✓ CONE (yellow)')
                        labeled = True
                        idx += 1
                    elif key == 'u':
                        self.labels[filename] = {'is_cone': True, 'color': 'unknown'}
                        print('✓ CONE (unknown)')
                        labeled = True
                        idx += 1
                    elif key == 'n':
                        self.labels[filename] = {'is_cone': False, 'color': None}
                        print('✓ NOT CONE')
                        labeled = True
                        idx += 1
                    elif key == 's':
                        print('⊘ SKIPPED')
                        labeled = True
                        idx += 1
                    elif key == 'q':
                        print('\nQUITTING...')
                        with open(self.output_json, 'w') as f:
                            json.dump(self.labels, f, indent=2)
                        print(f'Saved {len(self.labels)} labels')
                        return
                    else:
                        print(f'Invalid "{key}" >> ', end='', flush=True)
                
                rclpy.spin_once(self, timeout_sec=0.01)
            
            # Auto-save every 5
            if len(self.labels) % 5 == 0 and len(self.labels) > 0:
                with open(self.output_json, 'w') as f:
                    json.dump(self.labels, f, indent=2)
        
        with open(self.output_json, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f'\n✓ Done! {len(self.labels)} labels')


def main(args=None):
    rclpy.init(args=args)
    node = ClusterLabelerNode()
    node.run()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
