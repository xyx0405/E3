#!/bin/bash

# 1. 定义路径
PROJECT_ROOT="/code/E3-CryoFold"
DATA_SOURCE="/eds-storage/xyx/e3cryofold/data"
CONDA_ENV_NAME="e3cryofold"
SSH_BACKUP="/eds-storage/xyx/ssh_backup"

echo "🚀 开始初始化开发环境..."

# 2. 先恢复 SSH Key (必须在 git pull 之前！)
if [ ! -f "$HOME/.ssh/id_ed25519" ]; then
    echo "🔑 正在从持久盘恢复 SSH Key..."
    mkdir -p $HOME/.ssh
    if [ -f "$SSH_BACKUP/id_ed25519" ]; then
        cp $SSH_BACKUP/id_ed25519* $HOME/.ssh/
        chmod 600 $HOME/.ssh/id_ed25519
        echo "✅ SSH Key 恢复成功。"
    else
        echo "❌ 错误: 备份路径 $SSH_BACKUP 下找不到 Key，请手动生成并备份一次。"
    fi
fi

# 3. 检查并同步代码
echo "📦 检查代码同步状态..."
if [ -d ".git" ]; then
    # 尝试更新，如果失败可能是因为网络或Key不对
    git pull origin main || echo "⚠️ Git 拉取失败，请检查网络或 SSH Key。"
else
    echo "⚠️ 当前不是 Git 仓库，跳过拉取。"
fi

# 4. 建立数据软链接
echo "🔗 正在建立数据软链接..."
[ -L "./data" ] && rm ./data
if [ -d "$DATA_SOURCE" ]; then
    ln -s "$DATA_SOURCE" ./data
    echo "✅ 成功链接: ./data -> $DATA_SOURCE"
else
    echo "❌ 错误: 找不到数据源 $DATA_SOURCE"
fi

# 5. 初始化并激活 Conda (修复版)
echo "🐍 正在激活 Conda 环境: $CONDA_ENV_NAME"
# 自动寻找 conda.sh
CONDA_SHELL_PATH=$(conda info --base)/etc/profile.d/conda.sh
if [ -f "$CONDA_SHELL_PATH" ]; then
    source "$CONDA_SHELL_PATH"
    conda activate $CONDA_ENV_NAME
    echo "✅ 环境已激活: $(which python)"
else
    echo "❌ 找不到 conda.sh，尝试手动激活..."
    source activate $CONDA_ENV_NAME
fi

echo "------------------------------------------------"
echo "🎉 所有配置已就绪！"
echo "------------------------------------------------"