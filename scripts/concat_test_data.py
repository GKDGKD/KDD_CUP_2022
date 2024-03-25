import pandas as pd
import os
from tqdm import tqdm

def merge_files(input_dir, output_dir, output_file):
    # 获取infile目录下的所有文件
    infile_files = os.listdir(os.path.join(input_dir, 'infile'))

    # 创建一个空的DataFrame用于存储结果
    result_df = pd.DataFrame()

    print('Merging files...')
    for infile_file in tqdm(infile_files):
        # 构造对应的outfile文件名
        outfile_file = infile_file.replace('in.csv', 'out.csv')

        # 读取infile和outfile的数据
        infile_path = os.path.join(input_dir, 'infile', infile_file)
        outfile_path = os.path.join(input_dir, 'outfile', outfile_file)

        print(f'Reading {infile_path} and {outfile_path}...')
        breakpoint()
        infile_df = pd.read_csv(infile_path)
        outfile_df = pd.read_csv(outfile_path)

        # 连接两个DataFrame，假设它们具有相同的行数和顺序
        merged_df = pd.concat([infile_df, outfile_df], axis=0)

        # 将合并后的DataFrame添加到结果DataFrame中
        result_df = pd.concat([result_df, merged_df], ignore_index=True)

    # 将结果写入到输出文件
    output_path = os.path.join(output_dir, output_file)
    result_df.to_csv(output_path, index=False)
    print('Done.')


input_directory  = './data/test/final_phase_test'
output_directory = './data/test'
output_filename  = 'test.csv'

merge_files(input_directory, output_directory, output_filename)
