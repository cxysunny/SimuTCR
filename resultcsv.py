import re
import csv

log_path = "./results/tcr_pmhc/info.log"
output_path = "./results/tcr_pmhc/metrics.csv"

# 用于提取的正则表达式
test_line_pattern = re.compile(r"Test Loss: ([\d.]+), Test Auc: ([\d.]+)")
train_line_pattern = re.compile(r"Epoch (\d+), Average Loss: ([\d.]+)")

# 保存数据
metrics = {}

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        test_match = test_line_pattern.search(line)
        train_match = train_line_pattern.search(line)

        if train_match:
            epoch = int(train_match.group(1))
            train_loss = float(train_match.group(2))
            metrics.setdefault(epoch, {})["train_loss"] = train_loss

        elif test_match:
            test_loss = float(test_match.group(1))
            test_auc = float(test_match.group(2))
            # 假设test loss 总是跟在train之后一行
            if 'epoch' in locals():
                metrics.setdefault(epoch, {})["test_loss"] = test_loss
                metrics.setdefault(epoch, {})["test_auc"] = test_auc

# 写入 CSV 文件，保留4位小数
with open(output_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["epoch", "train_loss", "test_loss", "test_auc"])
    for epoch in sorted(metrics):
        row = metrics[epoch]
        writer.writerow([
            epoch,
            f"{row.get('train_loss', ''):.4f}" if "train_loss" in row else "",
            f"{row.get('test_loss', ''):.4f}" if "test_loss" in row else "",
            f"{row.get('test_auc', ''):.4f}" if "test_auc" in row else ""
        ])

print(f"Metrics saved to {output_path}")

