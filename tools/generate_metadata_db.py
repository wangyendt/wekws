#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库构建脚本 - Stage 1.5
作者: Wayne
功能: 将所有数据集的元数据整合到SQLite数据库，支持快速查询和筛选
"""

import os
import sys
import json
import sqlite3
import argparse
from pathlib import Path
from tqdm import tqdm


def create_database(db_path):
    """
    创建数据库和表结构
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建主表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audio_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            utt_id TEXT UNIQUE NOT NULL,
            dataset TEXT NOT NULL,
            split TEXT NOT NULL,
            label_type TEXT NOT NULL,
            
            -- 音频信息
            wav_path TEXT NOT NULL,
            duration REAL,
            text_content TEXT,
            
            -- 录音条件
            distance TEXT,
            angle TEXT,
            noise_volume TEXT,
            noise_type TEXT,
            
            -- 说话人信息
            gender TEXT,
            age INTEGER,
            speaker_id TEXT,
            
            -- 其他
            keyword_id INTEGER,
            
            -- 索引时间
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 创建索引以加速查询
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_utt_id ON audio_metadata(utt_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_split ON audio_metadata(split)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_gender ON audio_metadata(gender)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_age ON audio_metadata(age)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_distance ON audio_metadata(distance)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_noise_volume ON audio_metadata(noise_volume)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_noise_type ON audio_metadata(noise_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_label_type ON audio_metadata(label_type)')
    
    conn.commit()
    return conn


def load_json_metadata(json_path):
    """
    加载JSON元数据文件
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_data_list(data_list_path):
    """
    加载 data.list 文件（JSON Lines格式）
    """
    data_dict = {}
    with open(data_list_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data_dict[item['key']] = item
    return data_dict


def parse_age(age_str):
    """
    解析年龄字符串为整数
    """
    try:
        return int(age_str)
    except:
        return None


def parse_noise_volume(noise_volume_str):
    """
    解析噪声音量字符串（如 "00db" -> 0, "10db" -> 10）
    """
    try:
        return noise_volume_str.replace('db', '')
    except:
        return noise_volume_str


def insert_metadata_batch(conn, metadata_list):
    """
    批量插入元数据
    """
    cursor = conn.cursor()
    
    cursor.executemany('''
        INSERT OR REPLACE INTO audio_metadata 
        (utt_id, dataset, split, label_type, wav_path, duration, text_content,
         distance, angle, noise_volume, noise_type, gender, age, speaker_id, keyword_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', metadata_list)
    
    conn.commit()


def process_split(conn, project_dir, split_name, dataset_name='mobvoi_hotword'):
    """
    处理单个数据集分割（train/dev/test）
    """
    print(f"\n处理 {split_name} 数据集...")
    
    # 路径
    resources_dir = os.path.join(project_dir, 'data/mobvoi_hotword_dataset_resources')
    data_list_path = os.path.join(project_dir, f'data/{split_name}/data.list')
    
    # 加载 data.list（包含duration和text）
    print(f"  加载 data.list...")
    data_dict = load_data_list(data_list_path)
    print(f"  ✅ 加载了 {len(data_dict)} 条记录")
    
    # 加载元数据JSON文件
    metadata_all = []
    
    for label_type in ['p', 'n']:  # p=positive, n=negative
        json_path = os.path.join(resources_dir, f'{label_type}_{split_name}.json')
        
        if not os.path.exists(json_path):
            print(f"  ⚠️  跳过 {json_path}（文件不存在）")
            continue
        
        print(f"  加载 {label_type}_{split_name}.json...")
        metadata_json = load_json_metadata(json_path)
        print(f"  ✅ 加载了 {len(metadata_json)} 条元数据")
        
        # 合并数据
        metadata_batch = []
        missing_count = 0
        
        for item in tqdm(metadata_json, desc=f"  处理 {label_type}_{split_name}"):
            utt_id = item['utt_id']
            
            # 从 data.list 中获取 duration 和 text
            data_info = data_dict.get(utt_id, {})
            
            if not data_info:
                missing_count += 1
                continue
            
            # 准备插入数据
            metadata_batch.append((
                utt_id,
                dataset_name,
                split_name,
                'positive' if label_type == 'p' else 'negative',
                data_info.get('wav', ''),
                data_info.get('duration', 0.0),
                data_info.get('txt', ''),
                item.get('distance', ''),
                item.get('angle', ''),
                parse_noise_volume(item.get('noise_volume', '')),
                item.get('noise_type', ''),
                item.get('gender', ''),
                parse_age(item.get('age', '')),
                item.get('speaker_id', ''),
                item.get('keyword_id', 0)
            ))
        
        # 批量插入
        if metadata_batch:
            insert_metadata_batch(conn, metadata_batch)
            print(f"  ✅ 插入了 {len(metadata_batch)} 条记录到数据库")
        
        if missing_count > 0:
            print(f"  ⚠️  有 {missing_count} 条记录在 data.list 中未找到")


def print_statistics(conn):
    """
    打印数据库统计信息
    """
    cursor = conn.cursor()
    
    print("\n" + "="*60)
    print("数据库统计信息")
    print("="*60)
    
    # 总记录数
    cursor.execute("SELECT COUNT(*) FROM audio_metadata")
    total_count = cursor.fetchone()[0]
    print(f"\n总记录数: {total_count:,}")
    
    # 按split统计
    print("\n按数据集分割统计:")
    cursor.execute("""
        SELECT split, COUNT(*) as count 
        FROM audio_metadata 
        GROUP BY split
        ORDER BY count DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:10s}: {row[1]:8,} 条")
    
    # 按label_type统计
    print("\n按标签类型统计:")
    cursor.execute("""
        SELECT label_type, COUNT(*) as count 
        FROM audio_metadata 
        GROUP BY label_type
        ORDER BY count DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:10s}: {row[1]:8,} 条")
    
    # 按性别统计
    print("\n按性别统计:")
    cursor.execute("""
        SELECT gender, COUNT(*) as count 
        FROM audio_metadata 
        WHERE gender IS NOT NULL AND gender != ''
        GROUP BY gender
        ORDER BY count DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:10s}: {row[1]:8,} 条")
    
    # 按年龄统计（前10）
    print("\n按年龄统计（前10）:")
    cursor.execute("""
        SELECT age, COUNT(*) as count 
        FROM audio_metadata 
        WHERE age IS NOT NULL
        GROUP BY age
        ORDER BY count DESC
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:3d} 岁: {row[1]:8,} 条")
    
    # 按距离统计
    print("\n按距离统计:")
    cursor.execute("""
        SELECT distance, COUNT(*) as count 
        FROM audio_metadata 
        WHERE distance IS NOT NULL AND distance != ''
        GROUP BY distance
        ORDER BY distance
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:10s}: {row[1]:8,} 条")
    
    # 按噪声音量统计
    print("\n按噪声音量统计:")
    cursor.execute("""
        SELECT noise_volume, COUNT(*) as count 
        FROM audio_metadata 
        WHERE noise_volume IS NOT NULL AND noise_volume != ''
        GROUP BY noise_volume
        ORDER BY noise_volume
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:10s}: {row[1]:8,} 条")
    
    # 按噪声类型统计
    print("\n按噪声类型统计:")
    cursor.execute("""
        SELECT noise_type, COUNT(*) as count 
        FROM audio_metadata 
        WHERE noise_type IS NOT NULL AND noise_type != ''
        GROUP BY noise_type
        ORDER BY count DESC
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]:15s}: {row[1]:8,} 条")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='构建音频元数据数据库（Stage 1.5）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python generate_metadata_db.py
  python generate_metadata_db.py --output-db ./metadata.db
  python generate_metadata_db.py --splits train dev test
        """
    )
    
    parser.add_argument(
        '--output-db',
        type=str,
        default=None,
        help='输出数据库路径（默认: data/metadata.db）'
    )
    
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'dev', 'test'],
        help='要处理的数据集分割（默认: train dev test）'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制重建数据库（删除已存在的数据库）'
    )
    
    args = parser.parse_args()
    
    # 获取项目目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    # 设置数据库路径
    if args.output_db:
        db_path = args.output_db
    else:
        db_path = os.path.join(project_dir, 'data/metadata.db')
    
    print("="*60)
    print("音频元数据数据库构建工具")
    print("="*60)
    print(f"项目目录: {project_dir}")
    print(f"数据库路径: {db_path}")
    print(f"处理数据集: {', '.join(args.splits)}")
    print("="*60)
    
    # 检查数据库是否已存在
    if os.path.exists(db_path):
        if args.force:
            print(f"\n⚠️  删除已存在的数据库: {db_path}")
            os.remove(db_path)
        else:
            print(f"\n⚠️  数据库已存在: {db_path}")
            response = input("是否要重建数据库？这将删除所有现有数据。(y/N): ")
            if response.lower() != 'y':
                print("取消操作。")
                sys.exit(0)
            os.remove(db_path)
    
    try:
        # 创建数据库
        print("\n📊 创建数据库结构...")
        conn = create_database(db_path)
        print("✅ 数据库创建成功")
        
        # 处理各个数据集分割
        for split in args.splits:
            process_split(conn, project_dir, split)
        
        # 打印统计信息
        print_statistics(conn)
        
        # 关闭数据库
        conn.close()
        
        print(f"\n✅ 数据库构建完成！")
        print(f"📁 数据库位置: {db_path}")
        print(f"💾 数据库大小: {os.path.getsize(db_path) / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
