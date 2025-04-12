import yaml
import re

# 正则表达式：
# 匹配以 "GSH" 或 "CSH" 开头，后面跟任意字符（非贪婪），直到遇到 "LENG" 或 "LKNG"
pattern = re.compile(r'(?:GSH|CSH).*?(?:LENG|LKNG)', re.IGNORECASE)

def truncate_sequence(seq: str) -> str:
    """
    截取序列中从 "GSH" 或 "CSH" 开始，到 "LENG" 或 "LKNG" 结束的子串。
    如果找不到匹配，则返回空字符串。
    """
    if not isinstance(seq, str):
        return ""
    m = pattern.search(seq)
    if m:
        return m.group(0)
    else:
        return ""

def main():
    input_file = "/home/xycui/project/af3_binding/mhc/hla_sequences.yml"
    output_file = "/home/xycui/project/af3_binding/mhc/hla_sequences_truncated.yml"
    
    # 读取原始 YAML 文件，假设数据为字典 key: HLA 标识，value: 序列字符串
    with open(input_file, "r") as f:
        data = yaml.safe_load(f)
    
    new_data = {}
    for key, seq in data.items():
        truncated = truncate_sequence(seq)
        if truncated:
            new_data[key] = truncated
        else:
            new_data[key] = ""
            print(f"Warning: {key} 未找到合适的截断区域")
    
    with open(output_file, "w") as f_out:
        yaml.safe_dump(new_data, f_out, sort_keys=False)

if __name__ == "__main__":
    main()

