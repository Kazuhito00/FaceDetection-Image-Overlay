# FaceDetection-Image-Overlay
顔検出を行い、検出した顔の上に画像を重ねるデモです。<br>
顔検出は[Star-Clouds/CenterFace](https://github.com/Star-Clouds/CenterFace)を利用しています。<br>
![nwgqx-oy39q](https://user-images.githubusercontent.com/37477845/99551738-8f268e00-29ff-11eb-8565-5dd9eaadc534.gif)

# Requirement 
* OpenCV 4.2.0 or later

# Demo
デモの実行方法は以下です。
```bash
python demo.py
```
デモ実行時には、以下のオプションが指定可能です。<br>
また、「image」ディレクトリの画像を差し替えることによって重畳画像を変更できます。<br>
（複数枚格納した場合はアニメーションを行い、1枚であれば固定画像となります）

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
画像重畳表示時の一辺の長さの切り上げ値<br>
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
* --use_model<br>
顔検出に使用するモデルの選択(centerface, dbfaceが使用可能)<br>
デフォルト：centerface

# Reference
* [Star-Clouds/CenterFace](https://github.com/Star-Clouds/CenterFace)
* [PINTO0309/PINTO_model_zoo/041_DBFace](https://github.com/PINTO0309/PINTO_model_zoo/tree/master/041_DBFace)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
FaceDetection-Image-Overlay is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).<br>
※Imageディレクトリ内の画像はMITライセンス対象外です

# License(CenterFace, DBFace)
CenterFace is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).<br>
DBFace is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
