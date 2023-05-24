import csv
import random
import datetime

def generate_data():
    data = []
    current_time = datetime.datetime.now()
    total_records = 1000
    
    node_to_tenant = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9]}  # Static mapping of Node to Tenant_ID
    tenant_to_shard = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8], 5: [9, 10], 6: [11, 12], 7: [13, 14], 8: [15, 16], 9: [17, 18]}  # Static mapping of Tenant_ID to Shard
    
    disk_utilization_base = 20.0  # Starting Disk Utilization value
    shard_size_base = 50.0  # Starting Shard Size value
    disk_utilization_increment = 1.0  # Disk Utilization increment per hour
    shard_size_increment = 2.0  # Shard Size increment per hour
    
    current_day = None
    data_growth = {}
    
    for _ in range(total_records):
        node = random.choice(list(node_to_tenant.keys()))
        tenant_id = random.choice(node_to_tenant[node])
        shard = random.choice(tenant_to_shard[tenant_id])
        
        time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        query_rate = random.uniform(0, 2000)
        
        # Calculate data growth based on overall disk utilization for a given node, tenant ID, and shard
        if current_day is None or current_time.day != current_day:
            current_day = current_time.day
            for nt in node_to_tenant.values():
                for node in nt:
                    for tenant in tenant_to_shard.keys():
                        if tenant in nt:
                            for shard in tenant_to_shard[tenant]:
                                key = f"{node}_{tenant}_{shard}"
                                disk_utilization = disk_utilization_base + ((_ - 1) * disk_utilization_increment)
                                data_growth[key] = random.uniform(-5, 5) * disk_utilization / 100.0
        
        key = f"{node}_{tenant_id}_{shard}"
        data_growth_rate = data_growth[key] if key in data_growth else 0.0
        
        cpu_utilization = random.uniform(0, 100)
        query_response_time = random.uniform(0, 1500)
        disk_utilization = disk_utilization_base + (_ * disk_utilization_increment)
        shard_size = shard_size_base + (_ * shard_size_increment)
        
        shard_split_required = "Yes" if ((query_rate > 1000 and query_response_time > 500 and data_growth_rate > 10)
                                         or (data_growth_rate > 10 and disk_utilization > 80 and shard_size > 1000)
                                         or (shard_size > 1000 and disk_utilization > 80)
                                         or (query_rate > 1000 and cpu_utilization > 80 and data_growth_rate > 10)
                                         or (cpu_utilization > 80 and query_response_time > 500 and data_growth_rate > 10)) else "No"
        
        data.append([time, node, tenant_id, shard, query_rate, data_growth_rate, cpu_utilization, disk_utilization,
                     query_response_time, shard_size, shard_split_required])
        
        current_time += datetime.timedelta(hours=1)  # Increment time by 1 hour
    
    random.shuffle(data)  # Shuffle the data
    
    return data

# Generate data
metrics_data = generate_data()

# Save data to CSV file
filename = "metrics_data.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Time", "Node", "Tenant_Id", "Shard", "Query Rate (QPS)", "Data Growth Rate (GB/day)",
                     "CPU Utilization (%)", "Disk Space Utilization (%)", "Query Response Time (ms)", "Shard Size (GB)",
                     "ShardSplit Required"])
    writer.writerows(metrics_data)

print(f"Data saved to {filename}.")
