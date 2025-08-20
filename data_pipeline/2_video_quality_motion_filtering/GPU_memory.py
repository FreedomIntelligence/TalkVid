from torch.cuda import memory_stats

stats = memory_stats(device=1)
allocated = stats.get("allocated_bytes.all.current", 0)
reserved = stats.get("reserved_bytes.all.current", 0)
print(f"Allocated: {allocated / 1e9:.2f} GB")
print(f"Reserved: {reserved / 1e9:.2f} GB")


