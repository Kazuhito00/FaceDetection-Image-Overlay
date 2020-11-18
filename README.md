# FaceDetection-Image-Overlay
![nwgqx-oy39q](https://user-images.githubusercontent.com/37477845/99551738-8f268e00-29ff-11eb-8565-5dd9eaadc534.gif)

# Requirement 
* OpenCV 4.2.0 or later

# Demo
デモの実行方法は以下です。
```bash
python demo.py
```
また、デモ実行時には、以下のオプションが指定可能です。

* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --ceil<br>
画像重畳表示時の一辺の切り上げ値<br>
デフォルト：50
* --image_ratio<br>
画像の一辺のサイズ補正<br>
デフォルト：1.2
* --x_offset<br>
顔へ重畳表示する際の画像のX座標オフセット<br>
デフォルト：0
* --y_offset<br>
顔へ重畳表示する際の画像のY座標オフセット<br>
デフォルト：-30

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
FaceDetection-Image-Overlay is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).<br>
※Imageディレクトリ内の画像はMITライセンス対象外です
