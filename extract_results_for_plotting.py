"""
从实验结果JSON文件中提取数据，用于绘图
"""

import json
import os
import glob

def extract_accuracy_from_json(json_path):
    """从JSON文件中提取准确率"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 提取学生分支的测试准确率（这是主要指标）
    stu_acc = data.get('stu_test_accTop1', 0)
    
    # 也提取集成准确率
    ensemble_acc = data.get('test_accTop1', 0)
    
    # 提取平均分支准确率
    mean_acc = data.get('mean_test_accTop1', 0)
    
    return {
        'student_acc': stu_acc,
        'ensemble_acc': ensemble_acc,
        'mean_acc': mean_acc,
        'epoch': data.get('epoch', -1)
    }

def find_best_results(base_dir='./CIFAR100/300/GL/'):
    """查找所有实验结果"""
    
    results = {}
    
    # 查找所有test_best_metrics开头的JSON文件
    pattern = os.path.join(base_dir, '**/test_best_metrics*.json')
    json_files = glob.glob(pattern, recursive=True)
    
    print(f"找到 {len(json_files)} 个结果文件\n")
    
    for json_file in json_files:
        filename = os.path.basename(json_file)
        
        # 从文件名中提取信息
        # 格式: test_best_metrics_[TYPE]_[GPU_ID]_[LAMBDA]_seed97div_[MODEL]_[DATASET]_[TIMESTAMP].json
        parts = filename.replace('.json', '').split('_')
        
        try:
            # 提取实验类型（相异度度量方式）
            if 'ensem' in filename:
                exp_type = 'ensemble_adaptive'
            elif 'cosine' in filename:
                exp_type = 'cosine'
            elif 'euclidean' in filename:
                exp_type = 'euclidean'
            elif 'kl' in filename:
                exp_type = 'kl'
            elif 'wasserstein1' in filename:
                exp_type = 'wasserstein1'
            elif 'wasserstein2' in filename:
                exp_type = 'wasserstein2'
            else:
                exp_type = 'unknown'
            
            # 提取lambda值和模型名
            lambda_val = None
            model_name = None
            for i, part in enumerate(parts):
                if i < len(parts) - 1 and parts[i].replace('.', '').isdigit():
                    lambda_val = float(parts[i])
                if 'resnet' in part or 'vgg' in part:
                    model_name = part
            
            # 读取准确率
            acc_data = extract_accuracy_from_json(json_file)
            
            key = f"{exp_type}_lambda{lambda_val}_{model_name}"
            results[key] = {
                'type': exp_type,
                'lambda': lambda_val,
                'model': model_name,
                'file': json_file,
                **acc_data
            }
            
            print(f"✓ {exp_type:20s} | Lambda: {lambda_val} | Student Acc: {acc_data['student_acc']:.2f}% | Epoch: {acc_data['epoch']}")
            
        except Exception as e:
            print(f"✗ 解析失败: {filename} - {e}")
    
    return results

def organize_results_by_metric(results):
    """按相异度度量方式组织结果"""
    
    organized = {}
    
    for key, data in results.items():
        metric = data['type']
        if metric not in organized:
            organized[metric] = []
        organized[metric].append(data)
    
    return organized

def generate_plotting_data(results):
    """生成绘图用的Python代码"""
    
    organized = organize_results_by_metric(results)
    
    print("\n" + "="*80)
    print("生成绘图数据代码:")
    print("="*80)
    
    # 提取所有相异度度量方式
    metrics = sorted(organized.keys())
    
    print("\n# 相异度度量方式")
    print(f"metrics = {metrics}")
    
    print("\n# 不同度量方式的准确率 (%)")
    print("accuracy_data = {")
    
    for metric in metrics:
        accs = [data['student_acc'] for data in organized[metric]]
        if accs:
            avg_acc = sum(accs) / len(accs)
            print(f"    '{metric}': {avg_acc:.2f},")
    
    print("}")
    
    print("\n" + "="*80)

def main():
    print("="*80)
    print("从实验结果中提取数据")
    print("="*80)
    print()
    
    # 查找结果文件
    results = find_best_results()
    
    if not results:
        print("\n❌ 没有找到实验结果文件")
        print("请确保已运行实验并生成了 test_best_metrics*.json 文件")
        return
    
    # 生成绘图数据
    generate_plotting_data(results)
    
    # 按度量方式分组显示
    organized = organize_results_by_metric(results)
    
    print("\n" + "="*80)
    print("按度量方式分组的结果:")
    print("="*80)
    
    for metric, data_list in sorted(organized.items()):
        print(f"\n{metric.upper()}:")
        for data in data_list:
            print(f"  - Lambda: {data['lambda']}, Student Acc: {data['student_acc']:.2f}%, File: {os.path.basename(data['file'])}")
    
    print("\n" + "="*80)
    print("提取完成！请将上述数据代码复制到绘图脚本中")
    print("="*80)

if __name__ == '__main__':
    main()


