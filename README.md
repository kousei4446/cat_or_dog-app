仮想環境の作成と有効化
他のPCでも仮想環境を作成し、有効化します。
python -m venv myenv
source myenv/bin/activate  # Mac/Linuxの場合
# myenv\Scripts\activate  # Windowsの場合
必要なパッケージのインストール
次に、requirements.txt を使ってプロジェクトの依存関係をインストールします。
pip install -r requirements.txt
