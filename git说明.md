# Git 上传代码到 GitHub 说明

> 本文档记录了将本地代码上传到 GitHub 仓库的完整步骤

---

## 1. 环境准备

### 1.1 安装 Git

从官网下载并安装 Git：https://git-scm.com/download/win

### 1.2 刷新环境变量（安装后终端无法识别 git 时使用）

```powershell
# 刷新 PowerShell 环境变量，使 git 命令可用
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# 验证 Git 是否安装成功
git --version
```

---

## 2. 初始化本地仓库

```powershell
# 进入项目目录
cd "F:\file\兼职\Deep\Deep"

# 初始化 Git 仓库（在项目根目录创建 .git 文件夹）
git init
```

---

## 3. 配置 Git 用户信息

```powershell
# 设置全局用户邮箱（用于提交记录）
git config --global user.email "12FM@users.noreply.github.com"

# 设置全局用户名（用于提交记录）
git config --global user.name "12FM"
```

---

## 4. 添加远程仓库

```powershell
# 查看当前远程仓库配置
git remote -v

# 移除旧的远程仓库（如果存在）
git remote remove origin

# 添加新的远程仓库
git remote add origin https://github.com/12FM/BIOE70077.git
```

---

## 5. 提交代码

```powershell
# 添加所有文件到暂存区（-A 表示所有文件，包括新增、修改、删除的文件）
git add -A

# 或者只添加当前目录下的所有文件
git add .

# 提交到本地仓库（-m 后面是提交信息）
git commit -m "Initial commit"
```

---

## 6. 推送到 GitHub

```powershell
# 将当前分支重命名为 main（GitHub 默认主分支名）
git branch -M main

# 推送到远程仓库（-u 设置上游分支，之后可以直接用 git push）
git push -u origin main

# 如果远程仓库已有内容需要覆盖，使用 --force 强制推送（谨慎使用）
git push -u origin main --force
```

---

## 7. 常见问题

### 7.1 权限问题（403 错误）

如果出现 `Permission denied` 或 `403` 错误，需要清除旧的凭据：

```powershell
# 删除 Windows 中保存的 GitHub 凭据
cmdkey /delete:git:https://github.com

# 重新推送（会弹出浏览器登录窗口）
git push -u origin main
```

### 7.2 查看仓库状态

```powershell
# 查看当前仓库状态（哪些文件被修改、添加、删除）
git status

# 查看提交历史
git log --oneline
```

### 7.3 更新 README 显示内容

GitHub 首页默认显示 `README.md` 文件内容：

```powershell
# 用其他文件替换 README.md（例如用中文说明文档）
Copy-Item "说明文档.md" "README.md" -Force

# 提交并推送更改
git add README.md
git commit -m "Update README"
git push origin main
```

---

## 8. 完整上传流程（一键复制）

```powershell
# 1. 进入项目目录
cd "F:\file\兼职\Deep\Deep"

# 2. 初始化仓库
git init

# 3. 配置用户信息
git config --global user.email "12FM@users.noreply.github.com"
git config --global user.name "12FM"

# 4. 添加远程仓库
git remote add origin https://github.com/12FM/BIOE70077.git

# 5. 添加所有文件
git add -A

# 6. 提交
git commit -m "Initial commit"

# 7. 设置主分支并推送
git branch -M main
git push -u origin main
```

---

## 9. 仓库地址

- **GitHub 仓库**：https://github.com/12FM/BIOE70077
