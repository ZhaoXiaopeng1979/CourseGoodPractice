### ch0 - 分词与词频统计

分词用到 jieba，词频统计用到 collections.Counter

#### jieba 分词

`jieba.cut()` 为一般分词，返回一个 generator；`jieba.posseg.cut()` 为词性标注分词。

三种分词模式：
* 精确模式 `jieba.cut(text, cut_all=False)`
* 全模式 `jieba.cut(text, cut_all=True)`
* 搜索引擎模式 `jieba.cut_for_search()`

我们选用 cut 方法默认的精确模式。

#### collections

Python 标准库 collections 里面提供了几种很好用的特殊数据类型，包括 dict 的子类 Counter。使用 Counter 做词频统计很方便，不用担心某个键不存在而引发 KeyError:

```python
cnt = Counter()
for word in words:
    cnt[word] += 1
print(cnt.mostcommon(10))
```


