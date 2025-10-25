#!/bin/bash
export PATH=/opt/conda/bin:$PATH

# 檢查是否有提供公鑰
if [ -z "$DEVELOPER_PUBLIC_KEY" ]; then
  echo "錯誤：未提供 DEVELOPER_PUBLIC_KEY 環境變數。"
  exit 1
fi

# 將環境變數中的公鑰寫入檔案
echo "$DEVELOPER_PUBLIC_KEY" > /home/developer/.ssh/authorized_keys

# 重新設定權限 (以防萬一)
chown developer:developer /home/developer/.ssh/authorized_keys
chmod 600 /home/developer/.ssh/authorized_keys

# 產生 SSH 伺服器金鑰
ssh-keygen -A

mkdir -p /run/sshd
chmod 755 /run/sshd

# 以 'exec' 啟動 SSH 伺服器，並保持在前景執行
echo "SSH 伺服器已啟動，準備接受連線..."
exec /usr/sbin/sshd -D