import yaml
import json
import os

def main():
    # 输入和输出路径
    input_yml = "/home/xycui/project/af3_binding/mhc/hla_sequences_truncated.yml"
    output_dir = "/home/xycui/project/af3_binding/mhc/mhcjson"
    
    # 如果输出目录不存在，则创建
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取 YAML 文件
    with open(input_yml, "r") as f:
        data = yaml.safe_load(f)
    
    # 遍历每个 HLA key，并生成对应的 JSON 文件
    # Python 3.7+ 默认 dict 保持插入顺序
    for idx, (hla_name, seq) in enumerate(data.items(), start=1):
        # 构造 JSON 数据结构
        json_data = {
            "name": hla_name,
            "modelSeeds": [1234],
            "sequences": [
                {
                    "protein": {
                        "id": "D",
                        "sequence": seq
                    }
                }
            ],
            "dialect": "alphafold3",
            "version": 1
        }
        output_file = os.path.join(output_dir, f"{idx}.json")
        # 写入 JSON 文件
        with open(output_file, "w") as out_f:
            json.dump(json_data, out_f, indent=4)
        print(f"生成 {output_file}")

if __name__ == "__main__":
    main()
