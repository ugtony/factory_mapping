#!/bin/bash
export PATH=/opt/conda/bin:$PATH

# 檢查是否有提供公鑰
if [ -z "$DEVELOPER_PUBLIC_KEY" ]; then
  echo "-----------------------------------------------------"
  echo "錯誤：未提供 DEVELOPER_PUBLIC_KEY 環境變數。"
  echo "無法啟動 SSH 服務。"
  echo "將改為啟動 Bash Shell，您可以手動設定 SSH。"
  echo "-----------------------------------------------------"
  exec bash
fi

# 將環境變數中的公鑰寫入 root 的 authorized_keys
echo "$DEVELOPER_PUBLIC_KEY" > /root/.ssh/authorized_keys

# 重新設定權限
chown root:root /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys

# 產生 SSH 伺服器金鑰
ssh-keygen -A

# 準備 SSHD 執行目錄
mkdir -p /run/sshd
chmod 755 /run/sshd

# 以 'exec' 啟動 SSH 伺服器，並保持在前景執行
echo "SSH 伺服器已啟動，準備接受 root 連線..."
exec /usr/sbin/sshd -D