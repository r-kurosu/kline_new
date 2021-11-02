### 実行方法

```python Height.py book/exp_height.csv``` 

### バックグランドで実行
```nohup python3 Height.py book/exp_height.csv > height.out &```

### サーバの接続方法
```ssh takeda@emily.co.mi.i.nagoya-u.ac.jp```

```ssh com15```

### ファイルの転送方法
```scp -r /Users/takedakiyoshi/lab/kline/KLINE takeda@emily.co.mi.i.nagoya-u.ac.jp:/home/takeda```


### ファイル指定
```scp /Users/takedakiyoshi/lab/kline/KLINE/height.py takeda@emily.co.mi.i.nagoya-u.ac.jp:/home/takeda/KLINE```

### サーバからローカルに転送
```scp takeda@emily.co.mi.i.nagoya-u.ac.jp:/home/takeda/KLINE/height.out /Users/takedakiyoshi/lab/kline/KLINE```

scp /Users/takedakiyoshi/lab/kline/KLINE/hybrid.py takeda@emily.co.mi.i.nagoya-u.ac.jp:/home/takeda/KLINE

