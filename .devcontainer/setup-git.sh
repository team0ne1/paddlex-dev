#!/bin/bash
# 环境：Linux bash (在容器内执行)

HOST_GITCONFIG="/tmp/host-gitconfig"

echo "========================================"
echo "🔄 开始同步宿主机 Git 身份信息..."
echo "========================================"

# 1. 检查宿主机 Git 配置文件是否成功挂载
if [ ! -f "$HOST_GITCONFIG" ]; then
    echo "⚠️ 警告: 未找到挂载的宿主机 Git 配置文件 ($HOST_GITCONFIG)"
    echo "提示: 容器将使用默认空白身份。如果需要提交代码，请手动配置 git config。"
    exit 0
fi

# 2. 从只读的宿主机配置中安全提取用户名和邮箱
# 使用 git config 原生命令读取，可以完美免疫 Windows CRLF 换行符带来的解析干扰
GIT_NAME=$(git config --file "$HOST_GITCONFIG" --get user.name)
GIT_EMAIL=$(git config --file "$HOST_GITCONFIG" --get user.email)

# 3. 将提取到的身份信息写入容器的全局环境 (~/.gitconfig)
if [ -n "$GIT_NAME" ] && [ -n "$GIT_EMAIL" ]; then
    git config --global user.name "$GIT_NAME"
    git config --global user.email "$GIT_EMAIL"
    echo "✅ 身份注入成功: $GIT_NAME <$GIT_EMAIL>"
else
    echo "⚠️ 警告: 宿主机配置中缺失 user.name 或 user.email"
fi

# 4. 强制清理或拦截 Windows 特定的凭证助手
# 这一步是为了防止任何意外带入的 credential.helper=manager 导致 Linux 容器报错。
# 清除后，凭证的自动无感转发将完全由 devcontainer 的 features/git:1 特性接管。
git config --global --unset credential.helper 2>/dev/null

echo "========================================"
echo "🚀 Git 环境初始化完毕！"
echo "========================================"