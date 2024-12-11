import os
import re

def remove_black_background_sentence(paragraph):
    # 按句子拆分段落
    sentences = re.split(r'(?<=\.)\s', paragraph)
    
    # 筛选出不包含 "black background" 的句子
    sentences_to_keep = [sentence for sentence in sentences if 'black background' not in sentence.lower()]
    
    # 返回被保留的句子和删除的句子
    sentences_deleted = len(sentences) - len(sentences_to_keep)
    return ' '.join(sentences_to_keep), sentences_deleted

# 输入目录和输出目录
input_dir = '/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K/text'
output_dir = '/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K/text_process'

# 确保输出目录存在，如果不存在则创建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 统计数据
total_files = 0
files_with_black_background = []
total_sentences_deleted = 0

# 遍历输入目录中的所有 txt 文件
for filename in os.listdir(input_dir):
    if filename.endswith('.txt'):
        total_files += 1
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)
        
        # 读取原文件内容
        with open(input_file_path, 'r', encoding='utf-8') as f:
            paragraph = f.read()
        
        # 处理段落
        processed_paragraph, sentences_deleted = remove_black_background_sentence(paragraph)
        
        # 统计信息
        if sentences_deleted > 0:
            files_with_black_background.append(filename)
        total_sentences_deleted += sentences_deleted
        
        # 将处理后的段落保存到新的文件中
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(processed_paragraph)

        print(f"Processed file saved as: {output_file_path}")

# 输出统计结果
print("\n=== 统计结果 ===")
print(f"总共处理的文件数量: {total_files}")
print(f"包含 'black background' 的文件: {files_with_black_background}")
print(f"总共删除了 {total_sentences_deleted} 个句子")
