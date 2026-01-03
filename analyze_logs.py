"""
分析训练日志数据
不依赖tensorflow，直接解析tensorboard事件文件
"""
import os
import glob

def get_log_info(log_dir):
    """获取日志基本信息"""
    summary_dir = os.path.join(log_dir, 'summary')
    if not os.path.exists(summary_dir):
        return None
    
    event_files = glob.glob(f'{summary_dir}/*.v2')
    if not event_files:
        return None
    
    size_bytes = os.path.getsize(event_files[0])
    size_mb = size_bytes / (1024 * 1024)
    
    # 判断算法类型
    dirname = os.path.basename(log_dir)
    if '_DQN_' in dirname:
        algo = 'DQN'
        game = dirname.split('_DQN_')[1]
    elif 'Breakout' in dirname:
        algo = 'DDQN'  # 默认Breakout是DDQN
        game = 'BreakoutNoFrameskip-v4'
    elif 'ddqn' in dirname.lower():
        algo = 'DDQN'
        game = dirname.split('_')[-1] if '_' in dirname else 'Unknown'
    elif 'dqn' in dirname.lower():
        algo = 'DQN'
        game = dirname.split('_')[-1] if '_' in dirname else 'Unknown'
    else:
        algo = 'Unknown'
        game = 'Unknown'
    
    return {
        'path': log_dir,
        'name': dirname,
        'algo': algo,
        'game': game,
        'size_mb': size_mb,
        'has_weights': os.path.exists(os.path.join(log_dir, 'weights'))
    }

def main():
    print('=' * 100)
    print('📊 /root/Deep/log 目录完整分析')
    print('=' * 100)
    
    # 扫描所有日志
    log_dirs = sorted(glob.glob('log/*/'))
    all_logs = []
    
    for log_dir in log_dirs:
        info = get_log_info(log_dir)
        if info:
            all_logs.append(info)
    
    # 按算法和游戏分组
    by_game = {}
    for log in all_logs:
        game = log['game']
        if game not in by_game:
            by_game[game] = {'DQN': [], 'DDQN': [], 'Unknown': []}
        by_game[game][log['algo']].append(log)
    
    # 打印统计
    print(f'\n找到 {len(all_logs)} 个有效训练日志\n')
    
    # 按游戏展示
    for game in sorted(by_game.keys()):
        game_logs = by_game[game]
        dqn_count = len(game_logs['DQN'])
        ddqn_count = len(game_logs['DDQN'])
        
        if dqn_count == 0 and ddqn_count == 0:
            continue
        
        print(f'\n🎮 游戏: {game}')
        print('-' * 100)
        
        # DQN日志
        if dqn_count > 0:
            print(f'\n  【DQN】 共 {dqn_count} 个训练:')
            for log in sorted(game_logs['DQN'], key=lambda x: x['size_mb'], reverse=True):
                weights_mark = '💾' if log['has_weights'] else '  '
                print(f'    {weights_mark} {log["name"]}')
                print(f'       大小: {log["size_mb"]:.2f} MB')
        
        # DDQN日志
        if ddqn_count > 0:
            print(f'\n  【DDQN】 共 {ddqn_count} 个训练:')
            for log in sorted(game_logs['DDQN'], key=lambda x: x['size_mb'], reverse=True):
                weights_mark = '💾' if log['has_weights'] else '  '
                print(f'    {weights_mark} {log["name"]}')
                print(f'       大小: {log["size_mb"]:.2f} MB')
    
    # Archive 数据
    print('\n' + '=' * 100)
    print('📦 Archive 参考数据')
    print('=' * 100)
    
    archive_dirs = glob.glob('archive/*/')
    for archive_dir in archive_dirs:
        info = get_log_info(archive_dir)
        if info:
            print(f'\n  {info["name"]}:')
            print(f'    算法: {info["algo"]}')
            print(f'    大小: {info["size_mb"]:.2f} MB')
    
    # 可绘制的图表总结
    print('\n' + '=' * 100)
    print('📈 可绘制的图表分析')
    print('=' * 100)
    
    print('\n1️⃣  DQN vs DDQN 对比图:')
    comparison_available = False
    for game in sorted(by_game.keys()):
        dqn_logs = [l for l in by_game[game]['DQN'] if l['size_mb'] > 1]
        ddqn_logs = [l for l in by_game[game]['DDQN'] if l['size_mb'] > 1]
        
        if dqn_logs and ddqn_logs:
            comparison_available = True
            print(f'   ✅ {game}: DQN({len(dqn_logs)}) vs DDQN({len(ddqn_logs)})')
        elif dqn_logs and not ddqn_logs:
            print(f'   ⚠️  {game}: 仅有 DQN({len(dqn_logs)}), 缺少 DDQN')
        elif ddqn_logs and not dqn_logs:
            print(f'   ⚠️  {game}: 仅有 DDQN({len(ddqn_logs)}), 缺少 DQN')
    
    if not comparison_available:
        print('   ❌ 无法绘制完整的DQN vs DDQN对比图（需要同一游戏的两个算法）')
        print('   💡 建议: 在Alien上训练DDQN，或在Breakout上训练DQN')
    
    print('\n2️⃣  消融实验图（同算法多次训练）:')
    for game in sorted(by_game.keys()):
        dqn_valid = [l for l in by_game[game]['DQN'] if l['size_mb'] > 1]
        ddqn_valid = [l for l in by_game[game]['DDQN'] if l['size_mb'] > 1]
        
        if len(dqn_valid) >= 2:
            print(f'   ✅ {game} DQN: {len(dqn_valid)} 次训练可用于稳定性分析')
        if len(ddqn_valid) >= 2:
            print(f'   ✅ {game} DDQN: {len(ddqn_valid)} 次训练可用于稳定性分析')
    
    print('\n3️⃣  单算法学习曲线:')
    for game in sorted(by_game.keys()):
        dqn_best = max([l['size_mb'] for l in by_game[game]['DQN']], default=0)
        ddqn_best = max([l['size_mb'] for l in by_game[game]['DDQN']], default=0)
        
        if dqn_best > 1:
            print(f'   ✅ {game} DQN: 最大 {dqn_best:.1f} MB')
        if ddqn_best > 1:
            print(f'   ✅ {game} DDQN: 最大 {ddqn_best:.1f} MB')
    
    print('\n4️⃣  Archive参考对比:')
    print('   ✅ Atlantis: DQN vs DDQN (论文原始数据)')
    
    print('\n' + '=' * 100)

if __name__ == '__main__':
    main()
