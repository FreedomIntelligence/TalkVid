import psutil
import GPUtil
import time
import threading
import json
import os
import logging
from datetime import datetime
from threading import Lock

class SystemMonitor:
    def __init__(self, output_dir, interval=1.0):
        self.interval = interval
        self.monitoring = False
        self.stats_lock = Lock()
        self.stats = {
            'cpu_usage': [],
            'memory_usage': [],
            'gpu_usage': [],
            'disk_io': []
        }
        self.output_dir = output_dir
        self.start_time = None
        self.target_process = None
        self.experiment_name = None
        

        self.last_io_counters = {}
        self.last_io_time = None
        
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'system_monitor.log')),
                logging.StreamHandler()
            ]
        )
        
        logging.getLogger("psutil").setLevel(logging.ERROR)
        
    def get_process_children(self, pid):
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            return [parent] + children
        except psutil.NoSuchProcess as e:
            logging.warning(f"Process {pid} not found: {e}")
            return []
        except Exception as e:
            logging.error(f"Error getting process children: {e}")
            return []
        
    def get_io_speeds(self, processes):
        current_time = time.time()
        current_counters = {}
        total_read_speed = 0
        total_write_speed = 0

        
        for p in processes:
            try:
                if p.is_running():
                    try:
                        io_counters = p.io_counters()
                        current_counters[p.pid] = {
                            'read_bytes': io_counters.read_bytes,
                            'write_bytes': io_counters.write_bytes,
                            'time': current_time
                        }
                    except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
                        continue
            except psutil.NoSuchProcess:
                continue

       
        if self.last_io_time is not None:
            time_diff = current_time - self.last_io_time
            if time_diff > 0:  
                for pid, current in current_counters.items():
                    if pid in self.last_io_counters:
                        last = self.last_io_counters[pid]
                        read_diff = current['read_bytes'] - last['read_bytes']
                        write_diff = current['write_bytes'] - last['write_bytes']
                        
                        
                        if read_diff >= 0:
                            total_read_speed += read_diff / time_diff
                        if write_diff >= 0:
                            total_write_speed += write_diff / time_diff

        
        self.last_io_counters = current_counters
        self.last_io_time = current_time

        return total_read_speed, total_write_speed

    def get_system_stats(self):
        if not self.target_process:
            return None
            
        processes = self.get_process_children(self.target_process.pid)
        if not processes:
            return None
            
        
        for p in processes:
            try:
                p.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logging.debug(f"Error accessing process stats: {e}")
                pass
                
        
        time.sleep(0.5)
            
        
        try:
            cpu_percent = sum(p.cpu_percent() for p in processes if p.is_running())
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logging.debug(f"Error getting CPU usage: {e}")
            cpu_percent = 0
        
        
        memory_bytes = 0
        for p in processes:
            try:
                if p.is_running():
                    memory_bytes += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                logging.debug(f"Error getting memory info: {e}")
                continue
                
        total_memory = psutil.virtual_memory().total
        memory_percent = (memory_bytes / total_memory) * 100
        
        try:
            gpus = GPUtil.getGPUs()
            gpu_usage = [{'id': gpu.id, 'load': gpu.load*100, 'memory_used': gpu.memoryUsed, 
                         'memory_total': gpu.memoryTotal} for gpu in gpus]
        except Exception as e:
            logging.error(f"Error getting GPU stats: {e}")
            gpu_usage = []
        
        
        read_speed, write_speed = self.get_io_speeds(processes)
        
        return {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'gpu_usage': gpu_usage,
            'disk_read_speed': read_speed, 
            'disk_write_speed': write_speed  
        }
    
    def monitor_thread(self):
        while self.monitoring:
            try:
                current_stats = self.get_system_stats()
                if current_stats is None:
                    self.monitoring = False
                    break
                    
                with self.stats_lock:
                    self.stats['cpu_usage'].append(current_stats['cpu_percent'])
                    self.stats['memory_usage'].append(current_stats['memory_percent'])
                    self.stats['gpu_usage'].append(current_stats['gpu_usage'])
                    self.stats['disk_io'].append({
                        'read_speed': current_stats['disk_read_speed'],
                        'write_speed': current_stats['disk_write_speed']
                    })
                
            except Exception as e:
                logging.error(f"Error in monitor thread: {e}")
                
            time.sleep(self.interval)
    
    def start_monitoring(self, process, experiment_name=None):
        self.target_process = process
        self.experiment_name = experiment_name
        self.monitoring = True
        self.start_time = datetime.now()
        self.monitor_thread = threading.Thread(target=self.monitor_thread)
        self.monitor_thread.daemon = True  
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            try:
                
                self.monitor_thread.join(timeout=5.0)
                if self.monitor_thread.is_alive():
                    logging.warning("Monitor thread did not terminate within timeout")
            except Exception as e:
                logging.error(f"Error stopping monitor thread: {e}")
        
        with self.stats_lock:
            if not self.stats['cpu_usage']:  
                return None
                
            try:
                
                avg_stats = {
                    'cpu_usage_avg': sum(self.stats['cpu_usage']) / len(self.stats['cpu_usage']),
                    'memory_usage_avg': sum(self.stats['memory_usage']) / len(self.stats['memory_usage']),
                    'disk_io_avg': {
                        'read_speed_avg': sum(s['read_speed'] for s in self.stats['disk_io']) / len(self.stats['disk_io']),
                        'write_speed_avg': sum(s['write_speed'] for s in self.stats['disk_io']) / len(self.stats['disk_io'])
                    }
                }
                
                
                if self.stats['gpu_usage'] and self.stats['gpu_usage'][0]:
                    gpu_avgs = {}
                    for gpu_id in range(len(self.stats['gpu_usage'][0])):
                        gpu_loads = [snapshot[gpu_id]['load'] for snapshot in self.stats['gpu_usage']]
                        gpu_mem = [snapshot[gpu_id]['memory_used'] for snapshot in self.stats['gpu_usage']]
                        gpu_avgs[f'gpu_{gpu_id}'] = {
                            'load_avg': sum(gpu_loads) / len(gpu_loads),
                            'memory_used_avg': sum(gpu_mem) / len(gpu_mem)
                        }
                    avg_stats['gpu_usage_avg'] = gpu_avgs
                
                
                timestamp = self.start_time.strftime('%Y%m%d_%H%M%S')
                output_file = os.path.join(self.output_dir, f'{self.experiment_name}_system_stats_{timestamp}.json')
                
                with open(output_file, 'w') as f:
                    json.dump({
                        'experiment_name': self.experiment_name,
                        'timestamp': timestamp,
                        'average_stats': avg_stats,
                        'detailed_stats': self.stats
                    }, f, indent=4)
                
                return avg_stats
                
            except Exception as e:
                logging.error(f"Error calculating and saving statistics: {e}")
                return None

def create_monitor(output_dir):
    return SystemMonitor(output_dir) 