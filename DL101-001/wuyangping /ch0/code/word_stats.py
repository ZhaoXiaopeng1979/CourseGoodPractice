# coding: utf-8
import jieba
import pandas
import numpy
import collections

#Read file
input_file = open('happiness.txt', 'r')
text = input_file.read().decode('utf-8')
input_file.close()

#Split words
segments = []
segs = jieba.cut(text)
for seg in segs:
    if len(seg)>1:
        segments.append(seg)

#Get statistics by dict
dict = {}
for key in segments:
    if key in dict:
        dict[key] = dict[key] + 1
    else:
        dict[key] = 1

sorted_list = sorted(dict.items(), key=lambda x:x[1], reverse=True)
export_list = sorted_list[:10]

for (x,y) in export_list:
    print x,y

#Get statistics by collections.Counter
cnt = collections.Counter()
for word in segments:
    cnt[word] += 1

top10 = cnt.most_common()[0:10]

for element in top10:
    print element[0], element[1]

#get word statitics via pandas

segmentDF = pandas.DataFrame({'segment':segments})

segStat = segmentDF.groupby(by=["segment"])    ["segment"].agg({"计数":numpy.size}).reset_index().sort_values(by=["计数"],
    ascending=False
    );

segStat.head(10)
